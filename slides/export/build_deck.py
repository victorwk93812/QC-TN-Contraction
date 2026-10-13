"""build_deck.py — assemble deck/index.html from deck/shell.html and the act files.

    python export/build_deck.py

Act files are concatenated in the order of ORDER; each holds a run of <section> slides.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DECK = ROOT / "deck"
ORDER = ["act1", "act2", "act3", "dd1", "dd2", "dd3", "dd4", "act4", "act5", "backup"]

# Manim clips as optional vertical sub-slides (press ↓) under the slide whose figure they animate.
CLIPS = {
    "s01": ("M5", "The ring contracts to one number", "data/F03.json, data/F04.json"),
    "s23": ("M1", "Column sweep against MinFill on a 12×12 grid", "data/M1_sweep.json, data/M1_minfill.json"),
    "s31": ("M2", "Nested dissection builds the separator tree", "data/F12.json, data/F14.json"),
    "s34": ("M3", "FlowCutter on the 7×7 grid", "data/F18.json"),
    "s38": ("M4", "Girvan–Newman's dendrogram becomes a contraction tree", "data/F20.json"),
}


def clip_section(sid, clip, title, src):
    return f"""<section id="{sid}-clip" class="clip-slide" data-title="{title}">
  <video class="clip" data-autoplay muted playsinline controls preload="auto" src="media/{clip}.mp4" poster="media/{clip}.png"></video>
  <aside class="notes">Optional clip {clip} (Manim, manim/scenes.py), drawn from the same traces as the slide above: {src}. Press ↑ to return.</aside>
</section>"""


def attach_clips(html: str) -> str:
    for sid, (clip, title, src) in CLIPS.items():
        m = re.search(rf'<section[^>]*\bid="{sid}"', html)
        if not m:
            continue
        start = m.start()
        end = html.index("</section>", start) + len("</section>")
        html = html[:start] + "<section>\n" + html[start:end] + "\n" + clip_section(sid, clip, title, src) + "\n</section>" + html[end:]
    return html


def main():
    shell = (DECK / "shell.html").read_text()
    parts = []
    for name in ORDER:
        f = DECK / "slides" / f"{name}.html"
        if f.exists():
            parts.append(f"<!-- ===== {name} ===== -->\n" + f.read_text())
    html = shell.replace("<!-- SLIDES -->", attach_clips("\n".join(parts)))
    (DECK / "index.html").write_text(html)
    print(f"deck/index.html: {html.count('<section')} sections from {len(parts)} files")


if __name__ == "__main__":
    main()
