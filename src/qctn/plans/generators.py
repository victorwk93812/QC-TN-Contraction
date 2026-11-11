"""The paper's seven plan generators, for any network from `qctn.tn`.

Every generator ends in the same QXTools step: an elimination order over the network's bonds (the vertices of
the line graph) becomes a pairwise plan with `order_to_contraction_plan` (`qctn.plans.qxtools`). They differ
in how the order is found:

| generator       | order from                                                              |
|-----------------|-------------------------------------------------------------------------|
| `mf`            | QXGraphDecompositions `min_fill` on the line graph (Python, exact port) |
| `gn`            | reverse Girvan–Newman removal order on the tensor graph (Python)        |
| `fc`            | FlowCutter PACE 2017 tree decomposition of the line graph               |
| `tamaki`        | Tamaki et al. PACE 2017 heuristic tree decomposition                     |
| `jdrasil`       | Jdrasil heuristic tree decomposition                                    |
| `minfill_mrs`   | Jégou–Kanso–Terrioux PACE 2017 MinFill heuristic tree decomposition     |
| `minfillbg_mrs` | the same authors' second PACE 2017 heuristic                            |

Tree decompositions become orders with QXGraphDecompositions `order_from_tree_decomposition`. If a solver
returns no decomposition (killed before its first one), the order falls back to `min_fill`, as
`flow_cutter_contraction_plan` does (QXTools `contraction_planning.jl`, l. 102–106).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from qctn.plans import pace
from qctn.plans import qxtools as qx
from qctn.plans.authors import GENERATORS

TD_SOLVERS = ("fc", "tamaki", "jdrasil", "minfill_mrs", "minfillbg_mrs")


@dataclass
class GeneratedPlan:
    generator: str
    steps: list[tuple[str, str, str]]
    order: list[int]  # line-graph labels (1-based bond positions)
    width: int | None  # treewidth of the line-graph order reported by the method (td width or min_fill degree)
    elapsed: float
    fallback: bool = False  # solver gave no decomposition; min_fill used
    meta: dict = field(default_factory=dict)


def generate(tn, generator: str, *, time_limit: float = 10.0, seed: int | None = None,
             lg: qx.LineGraph | None = None, gn_backend: str = "rustworkx", wait: bool = False) -> GeneratedPlan:
    """One plan. Solvers are stopped after `time_limit` seconds (default 10, as the release's notebooks call
    `flow_cutter_contraction_plan(tnc; time=10)`); `wait=True` lets minfill_mrs / minfillbg_mrs finish."""
    if generator not in GENERATORS:
        raise KeyError(f"{generator!r} is not one of {GENERATORS}")
    lg = lg or qx.line_graph(tn)
    t0 = time.time()
    meta: dict = {}
    fallback = False
    if generator == "mf":
        width, order = qx.min_fill(lg)
    elif generator == "gn":
        from qctn.plans.girvan_newman import gn_order

        order, width = gn_order(tn, lg, backend=gn_backend), None
        meta["backend"] = gn_backend
    else:
        r = pace.run(generator, lg.to_gr(), time_limit=time_limit, seed=seed, wait=wait)
        meta.update(solver_elapsed=round(r.elapsed, 3), stopped=r.stopped, returncode=r.returncode)
        td = qx.parse_td(r.td_text)
        if not qx.valid_td(td, lg):
            fallback = True
            width, order = qx.min_fill(lg)
        else:
            width, order = td.width, qx.order_from_tree_decomposition(td)
            meta["bags"] = td.n_bags
    if sorted(order) != list(range(1, len(lg.bonds) + 1)):
        raise RuntimeError(f"{generator}: order is not a permutation of the {len(lg.bonds)} bonds")
    steps = qx.order_to_contraction_plan(order, lg)
    return GeneratedPlan(generator, steps, order, width, time.time() - t0, fallback, meta)


# ---------------------------------------------------------------------------------------------------------
# Output: the authors' text format, and comparison helpers


def _index_repr(ix: tuple, dim: int) -> str:
    """An index as ITensors prints it, with id 0: ITensors' random ids can't be reproduced."""
    tag, plev = ix
    if tag == "v":
        return f"(dim={dim}|id=0)"
    p = "'" * plev if plev <= 3 else f"'{plev}"
    return f'(dim={dim}|id=0|"{tag}"){p}'


def to_authors_plan(tn, steps, name: str):
    """An `AuthorsPlan` for `steps` on `tn`: names, ranks and common indices as `contract_pair_rank!` prints
    them; the time column (a measured GPU time in the release) is 0.0 and index ids are 0."""
    from qctn.plans.authors import AuthorsPlan, Step

    live = {t: list(ix) for t, ix in tn.tensors.items()}
    out = []
    for a, b, c in steps:
        A, B = live.pop(a), live.pop(b)
        sb = set(B)
        common = [x for x in A if x in sb]
        sc = set(common)
        C = [x for x in A if x not in sc] + [x for x in B if x not in sc]
        live[c] = C
        idx = "Index[" + ", ".join(_index_repr(x, tn.dims[x]) for x in common) + "]"
        out.append(Step(a, b, c, len(A), len(B), len(C), idx, 0.0))
    return AuthorsPlan(name, out)


def leaf_sets(steps) -> frozenset:
    """The plan as a set of leaf sets (one per intermediate): equal for two plans with the same tree,
    whatever the step order and intermediate names."""
    leaves: dict[str, frozenset] = {}
    out = set()
    for a, b, c in steps:
        s = leaves.pop(a, frozenset([a])) | leaves.pop(b, frozenset([b]))
        leaves[c] = s
        out.add(s)
    return frozenset(out)
