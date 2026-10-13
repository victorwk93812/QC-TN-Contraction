"""lint_numbers.py — list digits typed into slide text instead of bound from a trace.

    python export/lint_numbers.py

Ignores speaker notes, math (\\( \\), \\[ \\]), tag attributes, ids, act labels, slide titles that
name slide numbers, and digits inside data-bind elements. Everything printed should be either
structural (e.g. "7×7 grid", "slides 1–6") or replaced by a data-bind.
"""
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Lint(HTMLParser):
    SKIP = {"aside", "script", "style"}

    def __init__(self):
        super().__init__()
        self.stack, self.hits, self.section = [], [], "?"

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "section":
            self.section = a.get("id", "?")
        void = tag in {"br", "img", "line", "circle", "rect", "polygon", "path", "input", "meta", "link"}
        if not void:
            self.stack.append((tag, "data-bind" in a, "act-label" in (a.get("class") or ""), "placeholder" in (a.get("class") or "")))

    def handle_endtag(self, tag):
        while self.stack:
            t = self.stack.pop()
            if t[0] == tag:
                break

    def handle_data(self, data):
        if any(t[0] in self.SKIP or t[1] or t[2] or t[3] for t in self.stack):
            return
        text = re.sub(r"\\\(.*?\\\)|\\\[.*?\\\]", " ", data, flags=re.S)
        for m in re.finditer(r"\d[\d.,]*", text):
            ctx = text[max(0, m.start() - 25): m.end() + 25].replace("\n", " ").strip()
            self.hits.append((self.section, m.group(), ctx))


def main():
    total = 0
    for f in sorted((ROOT / "deck" / "slides").glob("*.html")):
        p = Lint()
        p.feed(f.read_text())
        for sec, num, ctx in p.hits:
            print(f"{f.name}:{sec}: {num!r:8} … {ctx}")
            total += 1
    print(f"{total} typed numbers to review")


if __name__ == "__main__":
    sys.exit(main())
