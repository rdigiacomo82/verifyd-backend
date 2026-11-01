# VeriFYD Access Phase C: recovered baseline and next integration checkpoint

Date: 2026-11-01

## GitHub / deployment facts
- Parent source archive: commit `abe540cba97608d872c8d294451a542d54187d8e`.
- Both uploaded Phase A v0.1.0 and Phase B v0.2.0 ZIPs were compared to the repository: all 31 committed text files have identical Git blob hashes. Preview PNG files are not committed.
- The branch is intentionally isolated. Render web/worker/cron services deploy `main` (`ff6d74bf86e020a58b0cea29e743d8f53853ee5b` at time of inspection); VeriFYD Access is not deployed to Render.
- Protect working VeriFYD Circle email, messaging, calls, adaptive UI, authorization and persistent device sessions. No backend integration required for this local browser prototype.

## Reproduced checks
- Five Phase B JavaScript modules: syntax PASS.
- `tests/test_core.cjs`, `tests/test_gaze_core.cjs` (18 cases), `tests/test_background.cjs`: PASS.
- `tests/test_browser.py`: PASS for simulated webpage controller.
- `tests/test_installed_extension.py`: blocked by sandbox browser administrative restriction (`ERR_BLOCKED_BY_ADMINISTRATOR`), **not** a pass.
- Physical real-camera Eye Gaze / blink recognition: **not yet validated**.
- Phase B ZIP intentionally excludes six MediaPipe model/runtime files, which must be installed locally before real-camera testing.

## Phase C acceptance sequence
1. Read-only Windows preflight: verify Phase B `extension/manifest.json`, eight source assets and six local MediaPipe assets. No privileged install required; do not alter Circle.
2. On Chrome or Edge install `extension/` using Developer mode / Load unpacked. Test keyboard, switching, pointer and pause before enabling camera.
3. In separate extension-origin camera popup, opt in to camera use, calibrate five points, validate direction, single/double blink, safety confirmation, and camera stop on exit/hidden tabs. Record device, distance, ambient lighting and accuracy.
4. iPad Pro 12.9-inch (5th gen): first verify Apple's native `Settings > Accessibility > Eye Tracking` controls Safari and other apps with gaze/dwell. This is distinct from VeriFYD's custom blink actions.
5. Independently prototype a Safari Web Extension packaged inside a separate iOS container, without touching the existing VeriFYD Circle app or its paused keyboard extension.
6. Desktop OS-wide cursor control is a later **native companion**, not a feature a browser extension can provide by itself. Android system-wide input likewise requires authorized platform accessibility services.

## Hard safety rules
- Keep gaze samples/video local; do not send frames to websites or the backend.
- Explicit activation and instant pause, with slow modes and switch fallback.
- No automatic purchase, delete, send, account-change, or sensitive actions.
- Do not merge this work into `main` or deploy Render services without physical-device testing and review.
- Check licensing and integrity of third-party MediaPipe assets before distribution.

See `tools/verifyd-access/phase-b-v0.2.0/README.md` and `NEXT_PHASE.md` for the original prototype instructions.