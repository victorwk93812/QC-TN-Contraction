"""The five tree-decomposition solvers behind `fc`, `tamaki`, `jdrasil`, `minfill_mrs`, `minfillbg_mrs`.

Built by `scripts/setup_solvers.sh` under `external/solvers/` (pinned commits there). Each reads a PACE 2017
`.gr` graph on stdin and writes a `.td` decomposition on stdout. FlowCutter, Tamaki and Jdrasil are anytime
solvers: they improve until SIGTERM and then print the best decomposition, so their result depends on the time
budget and on the machine. The two MinFill heuristics of Jégou, Kanso and Terrioux finish on their own and are
deterministic.

FlowCutter is called as QXGraphDecompositions' `flow_cutter` calls it (`treewidth.jl`, l. 42–87): graph on
stdin, SIGTERM after `time` seconds (Julia `kill`), seed 0 (`-s` is passed as `0` whatever the seed, and the
binary's default seed is also 0, `pace.cpp` l. 397). For the other anytime solvers the release states no
budget or seed; we use the same budget and the solvers' own default seeds (Tamaki's `tw-heuristic` script
passes `-s 42`; Jdrasil is unseeded without `-s`, and we pass `-s 42` as its README's example does).
"""
from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from qctn.paths import external_dir

JAVA_OPTS = ["-Xmx4g", "-Xss1g", "-XX:+UseSerialGC", "-XX:ActiveProcessorCount=1"]


def solvers_dir() -> Path:
    return external_dir() / "solvers"


def java() -> str:
    jdk = Path(os.environ.get("JDK", Path.home() / ".local/opt/jdk-21.0.13+9"))
    return str(jdk / "bin" / "java") if (jdk / "bin" / "java").exists() else "java"


def command(solver: str, seed: int | None = None) -> list[str]:
    s = solvers_dir()
    if solver == "fc":
        exe = [str(s / "flow-cutter-pace17" / "flow_cutter_pace17")]
        return exe + (["-s", str(seed)] if seed is not None else [])
    if solver == "tamaki":
        return [java(), *JAVA_OPTS, "-cp", str(s / "PACE2017-TrackA"), "tw.heuristic.MainDecomposer",
                "-s", str(42 if seed is None else seed)]
    if solver == "jdrasil":
        cmd = [java(), *JAVA_OPTS, "-cp", str(s / "Jdrasil" / "build" / "classes"), "jdrasil.Heuristic"]
        return cmd + ["-s", str(42 if seed is None else seed)]
    if solver in ("minfill_mrs", "minfillbg_mrs"):
        return [str(s / solver / solver)]
    raise KeyError(solver)


ANYTIME = {"fc", "tamaki", "jdrasil"}


def available(solver: str) -> bool:
    c = command(solver)
    if solver in ("tamaki", "jdrasil"):
        cp = Path(c[c.index("-cp") + 1])
        main = c[c.index("-cp") + 2].replace(".", "/") + ".class"
        return (cp / main).exists()
    return Path(c[0]).exists()


@dataclass
class SolverRun:
    solver: str
    td_text: str
    elapsed: float
    returncode: int | None
    stopped: bool  # sent SIGTERM at the budget


READ_DELAY = 1.0  # seconds between SIGTERM and reading the output


def run(solver: str, gr: str, time_limit: float = 10.0, seed: int | None = None,
        grace: float = 30.0, wait: bool = False, read_delay: float | None = READ_DELAY) -> SolverRun:
    """Run `solver` on the `.gr` text and send SIGTERM after `time_limit` seconds, as QXGraphDecompositions'
    `flow_cutter` does for FlowCutter (`sleep(time); kill(proc)`, then it reads the output file at once,
    `treewidth.jl` l. 62-65). The output is read `read_delay` seconds after the SIGTERM (None: when the process
    has exited). The release's plans indicate this for all five solvers (deviations.md, Phase 3): a solver that
    hasn't written its decomposition by then gives none, and the plan falls back to MinFill. `wait=True` lets
    the two non-anytime heuristics run to `time_limit + grace` instead."""
    with tempfile.TemporaryDirectory() as d:
        fin = Path(d) / "graph.gr"
        fin.write_text(gr)
        with open(fin) as stdin, open(Path(d) / "td.out", "w+") as out:
            t0 = time.time()
            p = subprocess.Popen(command(solver, seed), stdin=stdin, stdout=out, stderr=subprocess.DEVNULL,
                                 cwd=d, start_new_session=True)
            stopped, snapshot = False, None
            budget = time_limit + grace if (wait and solver not in ANYTIME) else time_limit
            try:
                p.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                stopped = True
                os.killpg(p.pid, signal.SIGTERM)
                if read_delay is not None:
                    try:
                        p.wait(timeout=read_delay)
                    except subprocess.TimeoutExpired:
                        pass
                    out.flush()
                    snapshot = Path(out.name).read_text()
                try:
                    p.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
            elapsed = time.time() - t0
            out.seek(0)
            text = out.read() if snapshot is None else snapshot
    return SolverRun(solver, text, elapsed, p.returncode, stopped)
