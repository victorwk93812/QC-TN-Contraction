"""Read and write the authors' plan files (`Data/plans` in their release).

One line per pairwise contraction, written by `contract_pair_rank!` and `contract_tn_rank_mock!` in
`src/funcions_article_IA.jl` (release 936630a):

    A B C rank(A) rank(B) rank(C)  <common indices, Julia repr> <seconds>

The two spaces before the index list and the Julia formatting of the time are kept, so writing a parsed
plan gives the file back byte for byte.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from qctn.paths import authors_plans

GENERATORS = ("fc", "gn", "jdrasil", "mf", "minfill_mrs", "minfillbg_mrs", "tamaki")

_LINE = re.compile(r"^(\S+) (\S+) (\S+) (\d+) (\d+) (\d+)  (Index(?:\{Int64\})?\[.*\]) (\S+)$")
# One ITensors index as printed: (dim=2|id=994|"Qubit28")''' or ...)'40 (prime level above 3).
_INDEX = re.compile(r'\(dim=(\d+)\|id=(\d+)(?:\|"([^"]*)")?\)(\'\d+|\'{0,3})')


class PlanFormatError(ValueError):
    pass


@dataclass(frozen=True)
class Index:
    """An ITensors index as it appears in the file. `id` is ITensors' random id modulo 1000."""
    dim: int
    id: int
    tags: str
    plev: int


@dataclass(frozen=True)
class Step:
    a: str
    b: str
    c: str
    rank_a: int
    rank_b: int
    rank_c: int
    indices: str  # the common indices exactly as printed
    time: float  # seconds of the (mock) GPU contraction, from `@timed`

    @property
    def n_common(self) -> int:
        """Number of contracted indices, as the release computes it: (rank_a + rank_b - rank_c) / 2."""
        k2 = self.rank_a + self.rank_b - self.rank_c
        if k2 % 2:
            raise PlanFormatError(f"odd rank difference in step {self.a} {self.b} {self.c}")
        return k2 // 2

    def parsed_indices(self) -> list[Index]:
        body = self.indices[self.indices.index("[") + 1:-1]
        out = []
        for m in _INDEX.finditer(body):
            p = m.group(4)
            plev = int(p[1:]) if p[1:].isdigit() else len(p)
            out.append(Index(int(m.group(1)), int(m.group(2)), m.group(3) or "", plev))
        return out


@dataclass
class AuthorsPlan:
    name: str  # file name without "resultats_" and ".txt", as in Features.xlsx
    steps: list[Step] = field(default_factory=list)

    @property
    def circuit(self) -> str:
        return split_name(self.name)[0]

    @property
    def generator(self) -> str | None:
        return split_name(self.name)[1]


def split_name(name: str) -> tuple[str, str | None]:
    """'dj_indep_qiskit_40_pla_fc' -> ('dj_indep_qiskit_40', 'fc'); names without '_pla_' -> (name, None)."""
    if "_pla_" in name:
        circuit, gen = name.rsplit("_pla_", 1)
        return circuit, gen
    return name, None


def plan_name(path: Path) -> str:
    """Same rule as `extraure_nom_circuit` in Funcions_de_Features_v_5.jl."""
    n = Path(path).name
    if n.startswith("resultats_") and n.endswith(".txt"):
        return n[len("resultats_"):-len(".txt")]
    return n


def julia_float(x: float) -> str:
    """Julia's `print(::Float64)`: shortest round-trip digits, `e` notation outside [1e-4, 1e16)."""
    if x == 0 or math.isinf(x) or math.isnan(x):
        return {0.0: "0.0"}.get(x, repr(x).replace("inf", "Inf").replace("nan", "NaN"))
    r = repr(x)
    if "e" in r:
        mant, exp = r.split("e")
        if "." not in mant:
            mant += ".0"
        return f"{mant}e{int(exp)}"
    if abs(x) >= 1e16:  # pragma: no cover - repr already uses e here
        return r
    return r if "." in r else r + ".0"


def parse_line(line: str) -> Step:
    m = _LINE.match(line.rstrip("\n"))
    if not m:
        raise PlanFormatError(f"malformed line: {line!r}")
    a, b, c, ra, rb, rc, idx, t = m.groups()
    return Step(a, b, c, int(ra), int(rb), int(rc), idx, float(t))


def format_line(s: Step) -> str:
    return f"{s.a} {s.b} {s.c} {s.rank_a} {s.rank_b} {s.rank_c}  {s.indices} {julia_float(s.time)}"


def parse_plan(text: str, name: str) -> AuthorsPlan:
    """Parse a plan file's text. Raises PlanFormatError with the line number of the first malformed line."""
    plan = AuthorsPlan(name)
    for no, line in enumerate(text.splitlines(), 1):
        try:
            plan.steps.append(parse_line(line))
        except PlanFormatError as e:
            raise PlanFormatError(f"resultats_{name}.txt:{no}: {e}") from None
    return plan


def read_plan(path: Path) -> AuthorsPlan:
    path = Path(path)
    return parse_plan(path.read_text(), plan_name(path))


def format_plan(plan: AuthorsPlan) -> str:
    return "".join(format_line(s) + "\n" for s in plan.steps)


def write_plan(plan: AuthorsPlan, path: Path) -> None:
    Path(path).write_text(format_plan(plan))


def plan_files(directory: Path | None = None) -> list[Path]:
    return sorted(Path(directory or authors_plans()).glob("resultats_*.txt"))


def read_all(directory: Path | None = None, texts: dict | None = None
             ) -> tuple[dict[str, AuthorsPlan], dict[str, str]]:
    """All plans in the release: (plans by name, parse errors by name). Pass a dict as `texts` to also
    collect each file's text by name."""
    plans, errors = {}, {}
    for p in plan_files(directory):
        name, text = plan_name(p), p.read_text()
        if texts is not None:
            texts[name] = text
        try:
            plans[name] = parse_plan(text, name)
        except PlanFormatError as e:
            errors[name] = str(e)
    return plans, errors
