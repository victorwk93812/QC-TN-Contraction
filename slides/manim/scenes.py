"""scenes.py — Manim clips M1–M5. Every scene reads the same data/*.json trace as its web figure;
nothing is recomputed here except display state (which vertices are alive, which region is shown).

    .venv/bin/manim -r 1920,1080 --fps 30 manim/scenes.py M1Race          (or M2NDTree, M3FlowCutter, M4GirvanNewman, M5Ring)
    sh export/render_clips.sh                                              (all five → deck/media/M*.mp4 + poster PNGs)
"""
from __future__ import annotations

import json
from pathlib import Path

from manim import (DOWN, LEFT, ORIGIN, RIGHT, UP, WHITE, AnimationGroup, Circle, Create, Dot, FadeIn, FadeOut,
                   Line, MathTex, Rectangle, RoundedRectangle, Scene, Square, Text, Transform, VGroup, Write,
                   config, np)

ACCENT, ACCENT_INK, ACCENT_SOFT = "#D85A30", "#993C1D", "#FAECE7"
DERIVED, DERIVED_INK, DERIVED_SOFT = "#7F77DD", "#534AB7", "#EEEDFE"
SIDE_B, SIDE_B_INK, SIDE_B_SOFT = "#1D9E75", "#0F6E56", "#E1F5EE"
MUTED, MUTED_2, INK, INK_2, SURFACE = "#B4B2A9", "#888780", "#1F1E1C", "#5F5E5A", "#F6F5F1"
FONT = "Source Sans 3"
DATA = Path(__file__).resolve().parent.parent / "data"
config.background_color = WHITE


def load(fid):
    return json.loads((DATA / f"{fid}.json").read_text())


def T(s, size=28, color=INK, weight="NORMAL"):
    return Text(s, font=FONT, font_size=size, color=color, weight=weight)


def placer(nodes, center, width, height):
    """Map trace coordinates (SVG units, y down) into a box of the Manim frame."""
    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    s = min(width / max(x1 - x0, 1e-9), height / max(y1 - y0, 1e-9))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return {v: np.array([center[0] + (x - cx) * s, center[1] - (y - cy) * s, 0]) for v, (x, y) in nodes.items()}


def edge_lines(trace, P, color=MUTED, width=1.5):
    return {frozenset((str(a), str(b))): Line(P[str(a)], P[str(b)], color=color, stroke_width=width) for a, b in trace["edges"]}

# ---------------------------------------------------------------- M1: sweep against MinFill


class M1Race(Scene):
    """Column sweep vs MinFill on a 12x12 grid: the live frontier (coral) and the bag sizes, side by side."""

    def construct(self):
        traces = [load("M1_sweep"), load("M1_minfill")]
        names = ["Column sweep", "MinFill"]
        centers = [(-3.6, 0.35), (3.6, 0.35)]
        title = T("Two orders on the same 12×12 grid", 34, weight="BOLD").to_edge(UP, buff=0.2)
        self.add(title)
        panels = []
        bmax = max(s["bag"] for t in traces for s in t["steps"])
        for t, name, c in zip(traces, names, centers):
            P = placer(t["nodes"], c, 4.2, 4.2)
            E = edge_lines(t, P, width=1.2)
            D = {v: Dot(P[v], radius=0.075, color=MUTED_2) for v in P}
            label = T(name, 30, weight="BOLD").next_to(VGroup(*D.values()), UP, buff=0.15)
            base = Line([c[0] - 2.4, -2.9, 0], [c[0] + 2.4, -2.9, 0], color=MUTED_2, stroke_width=1)
            ro = T("width 0", 26, INK_2).next_to(base, DOWN, buff=0.15)
            self.add(*E.values(), *D.values(), label, base, ro)
            panels.append({"t": t, "P": P, "D": D, "base": base, "ro": ro, "alive": set(P), "front": set(), "width": 0,
                           "bars": []})
        nsteps = len(traces[0]["steps"])
        bw = 4.8 / nsteps
        stride = 2
        for t0 in range(0, nsteps, stride):
            anims = []
            for p in panels:
                for t in range(t0, min(t0 + stride, nsteps)):
                    s = p["t"]["steps"][t]
                    v = str(s["elim"])
                    p["alive"].discard(v)
                    p["front"] = (p["front"] | {str(u) for u in s["nbrs"]}) - {v}
                    p["front"] &= p["alive"]
                    p["width"] = max(p["width"], len(s["nbrs"]))
                    h = 1.25 * s["bag"] / bmax
                    x = p["base"].get_start()[0] + (t + 0.5) * bw
                    bar = Rectangle(width=bw * 0.9, height=h, stroke_width=0, fill_color=MUTED, fill_opacity=1)
                    bar.move_to([x, -2.9 + h / 2, 0])
                    p["bars"].append((bar, s["bag"]))
                    anims.append(FadeIn(bar, run_time=0.01))
                for v, d in p["D"].items():
                    col = MUTED_2 if v in p["alive"] and v not in p["front"] else ACCENT if v in p["front"] else MUTED
                    op = 1.0 if v in p["alive"] else 0.18
                    if d.get_color().to_hex().upper() != col or abs(d.get_fill_opacity() - op) > 1e-3:
                        anims.append(d.animate.set_color(col).set_opacity(op))
            self.play(*anims, run_time=0.16)
            for p in panels:                  # readouts change instantly (no glyph morphing)
                p["ro"].become(T(f"width {p['width']}   frontier {len(p['front'])}", 26, INK_2).move_to(p["ro"].get_center()))
        for p in panels:                      # the widest bags in accent
            m = max(b for _, b in p["bars"])
            self.play(*[bar.animate.set_fill(ACCENT) for bar, b in p["bars"] if b == m], run_time=0.5)
        cost = [t["log2cost"] for t in traces]
        foot = T(f"log₂ cost: sweep {cost[0]:.2f}, MinFill {cost[1]:.2f}  ·  width: sweep {traces[0]['width']}, MinFill {traces[1]['width']}",
                 26, INK).to_edge(DOWN, buff=0.15)
        self.play(FadeIn(foot), run_time=0.6)
        self.wait(2)

# ---------------------------------------------------------------- M2: separator tree


class M2NDTree(Scene):
    """Nested dissection on 7x7: separators level by level; each new bag inherits boundary vertices from above."""

    def construct(self):
        t, f14 = load("F12"), load("F14")
        P = placer(t["nodes"], (-3.7, -0.2), 5.2, 5.2)
        E = edge_lines(t, P, width=1.2)
        D = {v: Dot(P[v], radius=0.11, color=MUTED_2) for v in P}
        self.add(T("Nested dissection builds a separator tree", 34, weight="BOLD").to_edge(UP, buff=0.3), *E.values(), *D.values())
        nbrs = {str(s["elim"]): [str(u) for u in s["nbrs"]] for s in t["steps"]}
        seps = [{"depth": s["depth"], "S": [str(v) for v in s["separator"]]} for s in t["separators"]]
        for s in seps:
            s["inherited"] = sorted(set().union(*(set(nbrs[v]) for v in s["S"])) - set(s["S"]))
            s["x"] = np.mean([P[v][0] for v in s["S"]])
            s["y"] = np.mean([P[v][1] for v in s["S"]])
        widest = max(l["bag"] for l in f14["levels"])
        ylev = [2.5, 0.9, -0.7]
        level_boxes = {}
        for d, lev in enumerate(f14["levels"]):
            group = sorted([s for s in seps if s["depth"] == d], key=lambda s: (s["x"], -s["y"]))
            n = len(group)
            xs = [3.4 + (i - (n - 1) / 2) * (6.4 / max(n, 1)) for i in range(n)]
            anims, boxes = [], []
            for s, x in zip(group, xs):
                anims += [D[v].animate.set_color(ACCENT).scale(1.35) for v in s["S"]]
                bcol = ACCENT if lev["bag"] == widest else DERIVED
                box = RoundedRectangle(width=1.45, height=0.9, corner_radius=0.12, stroke_color=bcol, stroke_width=2.5 if bcol == ACCENT else 2,
                                       fill_color=WHITE, fill_opacity=1).move_to([x, ylev[d], 0])
                txt = T(f"{lev['own']} + {lev['inherited']}" if d else f"{lev['own']}", 24).move_to(box.get_center() + UP * 0.15)
                sub = T(f"bag {lev['bag']}", 20, INK_2).move_to(box.get_center() + DOWN * 0.22)
                boxes.append((box, txt, sub, s))
            self.play(*anims, run_time=0.8)
            parents = level_boxes.get(d - 1, [])
            for box, txt, sub, s in boxes:
                links = []
                if parents:
                    par = min(parents, key=lambda b: abs(b[0].get_center()[0] - box.get_center()[0]))
                    links.append(Line(par[0].get_bottom(), box.get_top(), color=MUTED_2, stroke_width=1.5))
                self.play(*[Create(l) for l in links], FadeIn(box), run_time=0.35)
                if s["inherited"]:            # inherited boundary flows from the grid into the bag
                    ghosts = [Dot(P[v], radius=0.07, color=DERIVED) for v in s["inherited"]]
                    targets = [box.get_center() + LEFT * 0.55 + RIGHT * (1.1 * i / max(len(ghosts) - 1, 1)) + DOWN * 0.42
                               for i in range(len(ghosts))]
                    self.play(*[g.animate.move_to(tg).scale(0.6) for g, tg in zip(ghosts, targets)], run_time=0.9)
                    self.remove(*ghosts)
                self.play(Write(txt), FadeIn(sub), run_time=0.4)
            level_boxes[d] = boxes
        leaf = T(f"leaves: bags ≤ {f14['leaf_max_bag']}", 24, INK_2).move_to([3.4, -2.2, 0])
        foot = T(f"widest bag {max(l['bag'] for l in f14['levels'])}  ⇒  width {f14['width']}", 30, ACCENT_INK, "BOLD").to_edge(DOWN, buff=0.3).shift(RIGHT * 3.3)
        self.play(FadeIn(leaf), FadeIn(foot), run_time=0.6)
        self.wait(2)

# ---------------------------------------------------------------- M3: FlowCutter


class M3FlowCutter(Scene):
    """FlowCutter-lite on 7x7: source region (purple) and target region (teal) grow, each min cut (coral) is recorded."""

    def construct(self):
        t = load("F18")
        P = placer(t["nodes"], (-3.6, -0.3), 5.0, 5.0)
        E = edge_lines(t, P, width=1.2)
        D = {v: Dot(P[v], radius=0.13, color=MUTED_2) for v in P}
        self.add(T("FlowCutter: grow both sides, record every cut", 34, weight="BOLD").to_edge(UP, buff=0.3), *E.values(), *D.values())
        steps, pareto = t["steps"], set(t["pareto"])
        ox, oy, W, H = 1.2, -2.6, 5.4, 4.6
        xmax, ymax = max(s["size"] for s in steps) + 1, max(s["small_side"] for s in steps) + 2
        X = lambda v: ox + W * v / xmax
        Y = lambda v: oy + H * v / ymax
        axes = VGroup(Line([ox, oy, 0], [ox + W, oy, 0], color=MUTED_2), Line([ox, oy, 0], [ox, oy + H, 0], color=MUTED_2))
        xl = T("separator size |S|", 24, INK_2).next_to(axes[0], DOWN, buff=0.35)
        yl = T("smaller side", 24, INK_2).rotate(np.pi / 2).next_to(axes[1], LEFT, buff=0.6)
        ticks = VGroup(*[T(str(v), 20, INK_2).move_to([X(v), oy - 0.22, 0]) for v in range(0, xmax)])
        self.play(Create(axes), FadeIn(xl), FadeIn(yl), FadeIn(ticks), run_time=0.6)
        src, tgt = t["meta"]["source"], t["meta"]["target"]
        self.play(D[src].animate.set_color(DERIVED).scale(1.3), D[tgt].animate.set_color(SIDE_B).scale(1.3), run_time=0.4)
        ro = T("source and target corners", 26, INK_2).move_to([3.9, 2.6, 0])
        self.add(ro)
        yt = VGroup(*[T(str(v), 20, INK_2).move_to([ox - 0.25, Y(v), 0]) for v in range(0, ymax + 1, 5)])
        self.add(yt)
        front_pts = []
        for i, s in enumerate(steps):
            S, Tt, C = set(s["S"]), set(s["T"]), set(s["cut"])
            A, B = set(s["A"]), set(s["B"])
            anims = []
            for v, d in D.items():
                if v in C:
                    col = ACCENT
                elif v in S or (s["grew"] == "S" and v in A):
                    col = DERIVED
                elif v in Tt or (s["grew"] == "T" and v in B):
                    col = SIDE_B
                else:
                    col = MUTED_2
                if d.get_color().to_hex().upper() != col:
                    anims.append(d.animate.set_color(col))
            on = i in pareto
            pt = Dot([X(s["size"]), Y(s["small_side"]), 0], radius=0.09 if on else 0.05, color=ACCENT if on else MUTED_2)
            anims.append(FadeIn(pt))
            if on:
                front_pts.append(pt.get_center())
                if len(front_pts) > 1:
                    anims.append(Create(Line(front_pts[-2], front_pts[-1], color=ACCENT, stroke_width=3)))
            self.play(*anims, run_time=0.45)
            ro.become(T(f"cut {s['size']}  ·  smaller side {s['small_side']} of {t['n']}", 26, INK_2).move_to([3.9, 2.6, 0]))
        self.wait(2)

# ---------------------------------------------------------------- M4: Girvan–Newman


class M4GirvanNewman(Scene):
    """Girvan–Newman removes high-betweenness bonds; its dendrogram, read bottom-up, is a contraction tree."""

    def construct(self):
        t = load("F20")
        P = placer(t["nodes"], (-3.9, 0.2), 5.6, 3.0)
        E = edge_lines(t, P, width=2)
        Sq = {v: VGroup(Square(0.5, color=MUTED_2, fill_color=SURFACE, fill_opacity=1, stroke_width=1.5),
                        T(v, 20)).move_to(P[v]) for v in P}
        self.add(T("Girvan–Newman: split where shortest paths crowd", 34, weight="BOLD").to_edge(UP, buff=0.3), *E.values(), *Sq.values())
        rec = t["tree"]
        TP = placer({n["id"]: [n["x"], n["y"]] for n in rec["nodes"]}, (3.4, -0.2), 5.6, 5.2)
        byid = {n["id"]: n for n in rec["nodes"]}
        leaves = {n["id"]: VGroup(Square(0.42, color=MUTED_2, fill_color=SURFACE, fill_opacity=1, stroke_width=1.2),
                                  T(n["label"], 17)).move_to(TP[n["id"]]) for n in rec["nodes"] if n["kind"] == "leaf"}
        self.play(*[FadeIn(l) for l in leaves.values()], run_time=0.5)
        nodes = {}
        ro = T("every bond present", 24, INK_2).move_to([-3.9, -2.2, 0])
        self.add(ro)
        for i, s in enumerate(t["steps"], start=1):
            k = frozenset(s["removed"])
            line = E[k]
            anims = [line.animate.set_color(ACCENT).set_stroke(width=5)]
            self.play(*anims, run_time=0.18)
            anims = [line.animate.set_opacity(0.08).set_stroke(width=2, color=MUTED)]
            comps = s["components"]
            for v, sq in Sq.items():
                col = DERIVED if len(comps) > 1 and any(v in c and "A0" in c for c in comps) else SIDE_B if len(comps) > 1 and any(v in c and "B0" in c for c in comps) else MUTED_2
                anims.append(sq[0].animate.set_stroke(color=col))
            for n in rec["nodes"]:
                if n["kind"] != "leaf" and n.get("split_step") == i:
                    dot = Dot(TP[n["id"]], radius=0.1, color=MUTED_2)
                    nodes[n["id"]] = dot
                    kids = [e["child"] for e in rec["edges"] if e["parent"] == n["id"]]
                    anims += [FadeIn(dot)] + [Create(Line(TP[n["id"]], TP[c], color=MUTED_2, stroke_width=1.5)) for c in kids]
            self.play(*anims, run_time=0.35)
            ro.become(T(f"removed {s['removed'][0]}–{s['removed'][1]} · components {len(comps)} · Q = {s['modularity']:.3f}",
                        24, INK_2).move_to([-3.9, -2.2, 0]))
        note = T("read bottom-up: each split is one pairwise contraction", 26, DERIVED_INK).move_to([0, -3.4, 0])
        self.play(FadeIn(note), run_time=0.5)
        order = [n["id"] for n in rec["nodes"] if n["kind"] != "leaf"]      # post-order = contraction order
        mx = rec["max_rank"]
        for nid in order:
            n = byid[nid]
            col = ACCENT if n["rank"] == mx and n["kind"] == "internal" else DERIVED
            badge = VGroup(Circle(0.2, color=col, fill_color=WHITE, fill_opacity=1, stroke_width=2.5), T(str(n["rank"]), 18)).move_to(TP[nid])
            self.play(Transform(nodes[nid], badge), run_time=0.3)
        foot = T(f"GN tree: max rank {rec['max_rank']}, cost {rec['cost']}  ·  optimum: rank {t['opt_max_rank']}, cost {t['opt_cost']}",
                 26, INK).to_edge(DOWN, buff=0.12)
        self.play(FadeOut(note), FadeIn(foot), run_time=0.5)
        self.wait(2)

# ---------------------------------------------------------------- M5: the ring contracts to a scalar


class M5Ring(Scene):
    """Title animation: the four-tensor ring contracted pairwise, ((AB)(CD)), down to one number."""

    def construct(self):
        t = load("F03")
        tg = t["tensor_graph"]
        P = placer(tg["nodes"], (0, 0.1), 3.6, 3.6)
        box = lambda name, col=MUTED_2: VGroup(RoundedRectangle(width=0.8, height=0.8, corner_radius=0.1, color=col,
                                                                  fill_color=SURFACE, fill_opacity=1, stroke_width=2),
                                                T(name, 30, weight="BOLD"))
        B = {v: box(v).move_to(P[v]) for v in P}
        bonds = {lab: VGroup(Line(P[a], P[b], color=MUTED_2, stroke_width=3)) for a, b, lab in tg["bonds"]}
        labels = {lab: T(lab, 26, INK_2).move_to((P[a] + P[b]) / 2 + (np.array([0, 0.3, 0]) if abs(P[a][1] - P[b][1]) < 1e-6 else np.array([0.3, 0, 0]))) for a, b, lab in tg["bonds"]}
        self.play(*[Create(b) for b in bonds.values()], *[FadeIn(x) for x in B.values()], *[FadeIn(l) for l in labels.values()], run_time=1.2)
        f04 = load("F04")["trees"]["good"]
        steps = [n for n in f04["nodes"] if n["kind"] != "leaf"]
        # merge A·B (sum over i) and C·D (sum over k), then the two halves (sum over j, l)
        ab = box("AB", DERIVED).move_to((P["A"] + P["B"]) / 2)
        cd = box("CD", DERIVED).move_to((P["C"] + P["D"]) / 2)
        for (x, y, lab, new) in (("A", "B", "i", ab), ("C", "D", "k", cd)):      # sum over the shared index
            mid = new.get_center()
            self.play(B[x].animate.move_to(mid), B[y].animate.move_to(mid), FadeOut(bonds[lab]), FadeOut(labels[lab]),
                      bonds["j"].animate.set_opacity(0.25), bonds["l"].animate.set_opacity(0.25), run_time=0.8)
            self.remove(B[x], B[y])
            self.add(new)
        jl = VGroup(Line(ab.get_bottom() + LEFT * 0.2, cd.get_top() + LEFT * 0.2, color=MUTED_2, stroke_width=3),
                    Line(ab.get_bottom() + RIGHT * 0.2, cd.get_top() + RIGHT * 0.2, color=MUTED_2, stroke_width=3))
        jl_lab = VGroup(T("l", 26, INK_2).next_to(jl[0], LEFT, buff=0.12), T("j", 26, INK_2).next_to(jl[1], RIGHT, buff=0.12))
        r = T(f"rank {steps[0]['rank']}", 24, DERIVED_INK).next_to(ab, UP, buff=0.15)
        self.play(FadeOut(bonds["j"]), FadeOut(bonds["l"]), FadeOut(labels["j"]), FadeOut(labels["l"]),
                  Create(jl), FadeIn(jl_lab), FadeIn(r), run_time=0.8)
        scalar = Dot(ORIGIN, radius=0.22, color=ACCENT)
        self.play(ab.animate.move_to(ORIGIN), cd.animate.move_to(ORIGIN), FadeOut(jl), FadeOut(jl_lab), FadeOut(r), run_time=1.0)
        self.remove(ab, cd)
        self.play(FadeIn(scalar, scale=0.5), run_time=0.4)
        title = T("Finding good contraction orders", 48, weight="BOLD").next_to(scalar, DOWN, buff=0.8)
        sub = MathTex(r"((A\,B)(C\,D)) \;\to\; \text{one number}", color=INK).next_to(title, DOWN, buff=0.3)
        self.play(Write(title), FadeIn(sub), run_time=1.2)
        self.wait(2)
