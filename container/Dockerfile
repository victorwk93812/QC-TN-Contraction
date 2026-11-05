# Toolchain image for the render/export steps that bare metal lacks:
#   Node 22 + npm deps (reveal.js, KaTeX, fonts, DeckTape, Playwright), Chromium,
#   Manim 0.21 + ffmpeg + pango/cairo, TeX Live (report, Manim MathTex), Java (PACE solvers).
# Day-to-day Python work stays on the host .venv; see container/README.md.
# Built by .github/workflows/container.yml and pulled with: sh container/pull.sh
FROM node:22-bookworm

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      chromium fonts-liberation \
      ffmpeg libcairo2-dev libpango1.0-dev pkg-config build-essential \
      texlive-latex-base texlive-latex-recommended texlive-latex-extra \
      texlive-fonts-recommended texlive-science texlive-bibtex-extra \
      lmodern cm-super dvisvgm biber latexmk \
      default-jdk-headless git \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12 (Manim does not support 3.14) via a pinned uv.
COPY --from=ghcr.io/astral-sh/uv:0.8.0 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_NO_CACHE=1
COPY slides/requirements.txt /tmp/requirements.txt
RUN uv venv --python 3.12 /opt/venv \
 && uv pip install --python /opt/venv/bin/python -r /tmp/requirements.txt

# Node deps from the lockfile; browsers come from apt, so skip their downloads.
COPY slides/package.json slides/package-lock.json /opt/node/
RUN cd /opt/node && PUPPETEER_SKIP_DOWNLOAD=1 PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 \
      npm ci --no-audit --no-fund

ENV PATH=/opt/venv/bin:/opt/node/node_modules/.bin:$PATH \
    MANIM=/opt/venv/bin/manim \
    PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium
