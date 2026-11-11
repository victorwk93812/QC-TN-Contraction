#!/bin/sh
# User-local JDK for the PACE solvers (no root needed). Idempotent.
set -e
NAME=OpenJDK21U-jdk_x64_linux_hotspot_21.0.13_9.tar.gz
URL=https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.13%2B9/$NAME
SHA=d7f5d5cf865a2902b70ff549b97586a7d3f234e016f041ab447ea740c7d66d1c
DEST=$HOME/.local/opt/jdk-21.0.13+9
if [ ! -x "$DEST/bin/javac" ]; then
  mkdir -p "$HOME/.local/opt"
  TMP=$(mktemp -d)
  curl -fsSL -o "$TMP/$NAME" "$URL"
  echo "$SHA  $TMP/$NAME" | sha256sum -c -
  tar xzf "$TMP/$NAME" -C "$HOME/.local/opt"
  rm -rf "$TMP"
fi
"$DEST/bin/javac" -version
