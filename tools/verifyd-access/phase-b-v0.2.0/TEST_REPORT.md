# VeriFYD Access Phase B — test report

## Automated checks passed
- `node --check extension/background.js`
- `node --check extension/navigation-core.js`
- `node --check extension/gaze-core.js`
- `node --check extension/content.js`
- `node --check extension/gaze.js`
- `node tests/test_core.cjs` — Phase A navigation math
- `node tests/test_gaze_core.cjs` — face landmark extraction, five-point calibration, blink state machine
- `node tests/test_background.cjs` — authorized tab, isolated-world script injection, cryptographic session nonce, spoofed sensor message rejection, camera close
- `python tests/test_browser.py` — actual Chromium overlay integration: links/buttons, input keyboard, password denial, virtual pointer, camera-disabled browser demo, pause/close/reopen
- JSON parse of `manifest.json` — successful

## Test blocked by environment — not a code pass
- `python tests/test_installed_extension.py`: Chromium sandbox administration blocked navigation to even a local HTTP test page (`net::ERR_BLOCKED_BY_ADMINISTRATOR`). It is an optional environment-specific test and was **not** counted as passing.

## NOT tested yet
- Actual hardware camera stream from trusted extension popup
- MediaPipe model binaries (unavailable to download from the build environment)
- Gaze calibration quality on physical hardware (glasses, low-light, side-mounted cameras)
- True double blink recognition in a real live camera session
- Real Chrome/Edge unpacked-extension toolbar UI interactions
- Safari/iPad implementation and native desktop system-pointer control

## Suggested physical acceptance test
1. On a non-sensitive sample website, start extension and check all Phase A controls.
2. Install the pinned MediaPipe assets, open extension camera window, and verify the camera preview is visible only there.
3. Complete five-point calibration, try left/right/up/down movement at 60–90 cm, and repeat while moving farther away.
4. In Controls mode test five single blinks, five double blinks and two intentional pauses.
5. In Pointer mode test five ordinary buttons; confirm no unwanted double selections.
6. Test switching tabs, pressing Pause, closing the camera window, and denying camera permission — in all cases the camera should stop.
7. Verify that the visited site does not receive a camera permission prompt.
8. Repeat on Edge and Chrome; gather results for smoothing/sensitivity tuning before broader accessibility use.
