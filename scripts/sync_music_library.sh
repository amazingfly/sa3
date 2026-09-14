#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBRARY="$ROOT/musicLibrary"

mkdir -p "$LIBRARY/ogg"

find "$ROOT/outputs" -path "*/audio/*" \( -iname "*-combined-*.wav" -o -iname "*-combined-*.flac" \) -print0 |
while IFS= read -r -d '' src; do
  run_name="$(basename "$(dirname "$(dirname "$src")")")"
  base_name="$(basename "$src")"
  stem="${base_name%.*}"

  ogg_out="$LIBRARY/ogg/${run_name}__${stem}.ogg"

  if [ ! -f "$ogg_out" ]; then
    echo "Syncing $base_name to $ogg_out"
    ffmpeg -nostdin -y -v error -i "$src" -c:a libvorbis -q:a 6 "$ogg_out"
  fi
done

echo "Synced combined tracks to $LIBRARY/ogg"
