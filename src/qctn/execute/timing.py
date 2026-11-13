"""GPU timing of a compiled plan.

The release has no timing code for real contractions (see `src/notes/deviations.md`, Phase 4). The paper's
protocol: one warm-up run and 50 measured runs, the mean reported, Float32, `CUDA.synchronize` around each run,
with transfers, allocation, plan creation and JIT excluded. Here:

- leaves are cast to `dtype` (complex64 by default, Float32 parts) and copied to the GPU once, before timing;
- each run is the whole plan, `executor.run`, between two CUDA events on the current stream, with a
  `Device.synchronize()` before the start event and after the end event;
- `warmup` untimed runs first, then `repeats` timed ones; every repeat's time is returned, so the statistic
  (mean as in the paper, median, min) is chosen later;
- optionally `budget` seconds: no further repeats once the measured runs took that long in total (the count done
  is recorded). The paper treats a plan over 10 min, or out of memory, as failed; `max_warmup` stops a plan
  whose warm-up run is over a limit (status "slow").

Intermediates are allocated inside the timed region (from CuPy's memory pool, which after the warm-up holds the
blocks). Wall-clock time (`time.perf_counter` around the synchronised run) is recorded next to the event time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from qctn.execute.executor import Program, get_backend, prepare, run


@dataclass
class Timing:
    times: list[float] = field(default_factory=list)  # CUDA-event seconds per repeat (perf_counter on numpy)
    wall: list[float] = field(default_factory=list)  # perf_counter seconds per repeat, synchronised
    warmup: list[float] = field(default_factory=list)
    status: str = "ok"  # ok | oom | budget (fewer repeats than asked) | error: ...
    result: complex | None = None

    def stat(self, how: str = "mean") -> float:
        t = np.asarray(self.times)
        if not len(t):
            return float("nan")
        return float({"mean": np.mean, "median": np.median, "min": np.min}[how](t))


def time_program(program: Program, arrays: dict, *, backend: str = "cupy", dtype=np.complex64,
                 warmup: int = 1, repeats: int = 50, budget: float | None = None,
                 max_warmup: float | None = None) -> Timing:
    """`max_warmup`: if a warm-up run takes longer than this many seconds, stop with status "slow" and no
    measured repeats (the paper fails a plan over 10 minutes)."""
    xp = get_backend(backend)
    out = Timing()
    gpu = backend == "cupy"
    try:
        leaves = prepare(program, arrays, xp, dtype)
        if gpu:
            dev = xp.cuda.Device()
            start, end = xp.cuda.Event(), xp.cuda.Event()

        def once():
            if gpu:
                dev.synchronize()
                w0 = time.perf_counter()
                start.record()
                r = run(program, leaves, xp)
                end.record()
                end.synchronize()
                dev.synchronize()
                w = time.perf_counter() - w0
                return r, xp.cuda.get_elapsed_time(start, end) / 1e3, w
            w0 = time.perf_counter()
            r = run(program, leaves, xp)
            w = time.perf_counter() - w0
            return r, w, w

        r = None
        for _ in range(warmup):
            r, t, _w = once()
            out.warmup.append(t)
            if max_warmup is not None and t > max_warmup:
                out.status = "slow"
                repeats = 0
                break
        total = 0.0
        for _ in range(repeats):
            r, t, w = once()
            out.times.append(t)
            out.wall.append(w)
            total += t
            if budget is not None and total >= budget and len(out.times) < repeats:
                out.status = "budget"
                break
        if r is not None:
            v = r.get() if hasattr(r, "get") else np.asarray(r)
            out.result = complex(v.reshape(-1)[0]) if v.size == 1 else None
        del leaves, r
    except Exception as e:  # noqa: BLE001 - record and go on with the next plan
        name = type(e).__name__
        out.status = "oom" if "OutOfMemory" in name or isinstance(e, MemoryError) else f"error: {name}: {e}"
    finally:
        if gpu:
            xp.get_default_memory_pool().free_all_blocks()
    return out
