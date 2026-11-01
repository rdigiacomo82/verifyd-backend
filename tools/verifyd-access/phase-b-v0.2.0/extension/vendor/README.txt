This directory intentionally excludes large proprietary/third-party MediaPipe runtime and model assets.
The offline-at-runtime Eye Gaze feature requires one-time downloaded files in:
 vendor/vision_bundle.mjs
 vendor/wasm/vision_wasm_internal.js
 vendor/wasm/vision_wasm_internal.wasm
 vendor/wasm/vision_wasm_nosimd_internal.js
 vendor/wasm/vision_wasm_nosimd_internal.wasm
 vendor/face_landmarker.task
Run scripts/Install-MediaPipe-Assets.ps1 (Windows) or scripts/install-mediapipe-assets.sh (Mac/Linux), then reload the unpacked extension.
No downloads, frame uploads, or external inference occur when the extension runs.
