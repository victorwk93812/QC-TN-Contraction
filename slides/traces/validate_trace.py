"""validate_trace.py — check trace JSON files before any renderer reads them.

Usage: python validate_trace.py data/*.json
Exits non-zero on the first malformed file. Checks the schema, that every step refers to
live vertices, that fill edges join live neighbours, and that width/log2cost agree with steps.
"""
import json, math, sys


def check_tree(rec, err):
    """A drawable contraction tree from carving.tree_record."""
    ids = {n["id"] for n in rec["nodes"]}
    for n in rec["nodes"]:
        if n["rank"] != len(n["legs"]):
            err(f"tree node {n['id']}: rank != |legs|")
        if n["kind"] != "leaf" and n["exp"] != n["m"] + n["n"] + n["k"]:
            err(f"tree node {n['id']}: exp != m+n+k")
    for e in rec["edges"]:
        if e["parent"] not in ids or e["child"] not in ids:
            err("tree edge refers to a missing node")
    inter = [n["rank"] for n in rec["nodes"] if n["kind"] == "internal"] or [0]
    if max(inter) != rec["max_rank"]:
        err("tree max_rank mismatch")
    if sum(2 ** n["exp"] for n in rec["nodes"] if n["kind"] != "leaf") != rec["cost"]:
        err("tree cost mismatch")


def check_other(path, t, err):
    for key in ("schema", "id", "kind", "meta", "nodes", "edges"):
        if key not in t:
            err(f"missing key '{key}'")
    for v, p in t["nodes"].items():
        if not (isinstance(p, list) and len(p) == 2):
            err(f"node {v} needs [x, y]")
    nodes = set(t["nodes"])
    for a, b in t["edges"]:
        if nodes and (str(a) not in nodes or str(b) not in nodes):
            err(f"edge {a}-{b} refers to a missing node")
    if t["kind"] == "contraction_tree":
        recs = list(t.get("trees", {}).values()) + ([t["tree"]] if "tree" in t else [])
        for rec in recs:
            check_tree(rec, err)
    print(f"ok  {path}: {t['id']}, kind {t['kind']}")


def check(path):
    t = json.load(open(path))
    err = lambda m: sys.exit(f"{path}: {m}")
    if t.get("kind", "elimination") != "elimination":
        return check_other(path, t, err)
    for key in ("schema", "id", "nodes", "edges", "steps", "width", "log2cost"):
        if key not in t:
            err(f"missing key '{key}'")
    nodes = {str(v) for v in t["nodes"]}
    for v, p in t["nodes"].items():
        if not (isinstance(p, list) and len(p) == 2):
            err(f"node {v} needs [x, y]")
    alive = set(nodes)
    width = 0
    for i, s in enumerate(t["steps"]):
        v = str(s["elim"])
        if v not in alive:
            err(f"step {i}: {v} is not alive")
        nb = {str(u) for u in s["nbrs"]}
        if not nb <= alive - {v}:
            err(f"step {i}: neighbours not alive")
        for a, b in s["fill"]:
            if str(a) not in nb or str(b) not in nb:
                err(f"step {i}: fill edge {a}-{b} outside N(v)")
        if s["bag"] != len(nb) + 1:
            err(f"step {i}: bag != |N(v)|+1")
        width = max(width, len(nb))
        if s.get("width", width) != width:
            err(f"step {i}: running width mismatch")
        alive.discard(v)
    if alive:
        err(f"{len(alive)} vertices never eliminated")
    if width != t["width"]:
        err("width mismatch")
    bags = [s["bag"] for s in t["steps"]]
    m = max(bags)
    if abs(m + math.log2(sum(2 ** (b - m) for b in bags)) - t["log2cost"]) > 1e-9:
        err("log2cost mismatch")
    print(f"ok  {path}: {t['id']}, {len(t['steps'])} steps, width {width}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        check(p)
