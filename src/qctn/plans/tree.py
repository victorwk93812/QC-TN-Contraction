"""Contraction trees from pairwise plans, and conversion to cotengra.

A plan is a list of (a, b, c) steps on named tensors, as QXTools uses it: `a` and `b` are live tensors,
`c` is a fresh name for their product. The leaves are the names that are used before ever being produced.
cotengra is only the data structure here (`ContractionTree.from_path`); none of its optimisers are used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


class PlanError(ValueError):
    pass


@dataclass
class PlanTree:
    """A contraction forest. `children[c] = (a, b)`; `roots` are the tensors left at the end."""
    leaves: list[str]
    children: dict[str, tuple[str, str]]
    order: list[str]  # intermediate names in contraction order
    roots: list[str]

    @classmethod
    def from_steps(cls, steps: Iterable[tuple[str, str, str]], leaves: Sequence[str] | None = None):
        """Check that every step uses two live tensors and makes a new one.

        Without `leaves`, any unknown name used as an input is a leaf. With `leaves` (the network's
        tensor ids), inputs must be leaves or earlier results, and leaves never used become roots.
        """
        known = None if leaves is None else list(leaves)
        live: dict[str, None] = dict.fromkeys(known or [])
        seen = set(live)
        found_leaves: list[str] = list(known or [])
        children, order = {}, []
        for i, (a, b, c) in enumerate(steps):
            if a == b:
                raise PlanError(f"step {i}: {a} contracted with itself")
            for x in (a, b):
                if x not in live:
                    if x in seen:
                        raise PlanError(f"step {i}: {x} was already contracted")
                    if known is not None:
                        raise PlanError(f"step {i}: {x} is not a tensor of the network")
                    found_leaves.append(x)
                    seen.add(x)
                else:
                    del live[x]
            if c in seen:
                raise PlanError(f"step {i}: result name {c} is reused")
            seen.add(c)
            live[c] = None
            children[c] = (a, b)
            order.append(c)
        if known is None:
            roots = [c for c in order if c in live]
        else:
            roots = list(live)
        return cls(found_leaves, children, order, roots)

    @property
    def n_steps(self) -> int:
        return len(self.order)

    def ssa_path(self, leaf_order: Sequence[str] | None = None) -> list[tuple[int, int]]:
        """Static single assignment path: leaves are 0..n-1 in `leaf_order`, step i makes n + i."""
        leaf_order = list(leaf_order or self.leaves)
        pos = {x: i for i, x in enumerate(leaf_order)}
        for j, c in enumerate(self.order):
            pos[c] = len(leaf_order) + j
        return [(pos[a], pos[b]) for a, b in (self.children[c] for c in self.order)]

    def to_cotengra(self, inputs: Sequence[Sequence[str]], output: Sequence[str], size_dict: dict,
                    leaf_order: Sequence[str]):
        """A cotengra ContractionTree for this plan on a network given in `leaf_order`.

        Several roots (disconnected parts left at the end) are joined by outer products in the order
        the release does it, so the tree is complete.
        """
        import cotengra as ctg

        path = self.ssa_path(leaf_order)
        nxt = len(leaf_order) + len(path)
        ids = {x: i for i, x in enumerate(leaf_order)}
        ids.update({c: len(leaf_order) + j for j, c in enumerate(self.order)})
        rest = [ids[r] for r in self.roots]
        while len(rest) > 1:
            path.append((rest[0], rest[1]))
            rest = [nxt] + rest[2:]
            nxt += 1
        return ctg.ContractionTree.from_path(inputs, output, size_dict, ssa_path=path)
