#!/usr/bin/env bash
# For Mac or Linux. Downloads pinned MediaPipe ML assets before installing extension.
set -euo pipefail
BASE="$(cd "$(dirname "$0")/../extension/vendor" && pwd)"
mkdir -p "$BASE/wasm"
PACKAGE='https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22'
fetch(){ printf 'Fetching %s\n' "$1"; curl -fL --retry 2 --connect-timeout 15 --max-time 180 "$2" -o "$BASE/$1"; test "$(wc -c < "$BASE/$1")" -gt 4096; }
fetch vision_bundle.mjs "$PACKAGE/vision_bundle.mjs"
for fn in vision_wasm_internal.js vision_wasm_internal.wasm vision_wasm_nosimd_internal.js vision_wasm_nosimd_internal.wasm; do fetch "wasm/$fn" "$PACKAGE/wasm/$fn"; done
fetch face_landmarker.task 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
if command -v shasum >/dev/null 2>&1; then find "$BASE" -type f \( -name '*.js' -o -name '*.wasm' -o -name '*.mjs' -o -name '*.task' \) -exec shasum -a 256 {} \;; fi
printf 'Local model installed. Reload the unpacked extension in Chrome/Edge.\n'
