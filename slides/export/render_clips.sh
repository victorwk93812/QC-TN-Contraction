#!/bin/sh
# render_clips.sh — render Manim clips M1–M5 at 1080p30 into deck/media/, with a poster PNG (last frame)
# used by the deck before playback and by the PDF export.
set -e
cd "$(dirname "$0")/.."
OUT=deck/media
BUILD=export/out/manim
mkdir -p "$OUT" "$BUILD"
for pair in M1:M1Race M2:M2NDTree M3:M3FlowCutter M4:M4GirvanNewman M5:M5Ring; do
  id=${pair%%:*}; scene=${pair#*:}
  .venv/bin/manim -r 1920,1080 --fps 30 --disable_caching --media_dir "$BUILD" manim/scenes.py "$scene" > "$BUILD/$id.log" 2>&1
  src=$(find "$BUILD/videos" -path "*1080p30/$scene.mp4" | head -1)
  cp "$src" "$OUT/$id.mp4"
  ffmpeg -v error -y -sseof -0.2 -i "$OUT/$id.mp4" -frames:v 1 "$OUT/$id.png"
  echo "$id: $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT/$id.mp4") s, $(du -h "$OUT/$id.mp4" | cut -f1)"
done
