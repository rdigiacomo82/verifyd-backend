# VeriFYD Access — Phase B v0.2.0 (Camera Eye Gaze integration beta)

**Independent Chrome/Edge accessibility controller.** The VeriFYD Circle website, Render backend and paused Apple `VeriFYDAdaptiveKeyboard` Track A project remain unchanged.

## What was added to Phase A

- A dedicated **trusted extension-origin camera window**, opened only from a selected site after user action. This avoids giving arbitrary websites permission to use the camera.
- Locally processed Face Landmarker iris/eyelid signals (requires installing the model binaries as described below). Only normalized gaze coordinates + a blink boolean cross between extension components; **video frames are not uploaded, logged, or passed to visited pages**.
- Five-point calibration displayed over the current webpage, with rejection of extremely weak/noisy tracking.
- Smoothed, camera-driven **virtual webpage pointer** in Pointer mode. **Double blink = Select**. In Controls mode, **single blink = Next** and **double blink = Select**. Normal F7/F8/F9 and large-screen switch/keyboard controls remain available.
- Sensitive-action safeguard: gaze selection of controls labeled e.g. Delete, Send, Pay, Buy, Submit requires a *second* deliberate double-blink within six seconds. This is a heuristic, not a guarantee; do not test purchases, account deletion or sensitive forms with this beta.
- **Pause / Stop camera** terminates the stream; website navigation, window close, and camera-window visibility changes also stop it.
- No telemetry, no identity or VeriFYD Circle session required, no persistent camera permission granted to websites.

## IMPORTANT — complete one-time model installation before testing Eye Gaze

The download intentionally does **not** include Google MediaPipe's large third-party runtime/model files. The build environment could not fetch those files, so **live Eye Gaze is NOT ready immediately after unzip**. Phase A keyboard, scanning, pointer and buttons work without them.

**Windows (no administrator or Node needed):**

1. Extract this project ZIP to a normal folder.
2. Open PowerShell in that folder.
3. Run `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Install-MediaPipe-Assets.ps1`.
4. Confirm the script downloaded all six files and reported SHA256 digests. Internet is required **once for installation only**.

**Mac/Linux:** in a terminal inside the extracted folder run `bash scripts/install-mediapipe-assets.sh`.

The scripts obtain a pinned `@mediapipe/tasks-vision@0.10.22` build from jsDelivr and the official Google Face Landmarker model. The assets go **into `extension/vendor/`**, and the extension then runs the model entirely from its local files. Inspect downloaded files and their published license terms before distributing publicly.

## Load and test (Windows Chrome or Edge)

1. In Chrome open `chrome://extensions` (Edge: `edge://extensions`). Turn on Developer mode.
2. Click **Load unpacked**, select the **extension** folder (containing `manifest.json`), and use **Reload** if an older VeriFYD Access version is already installed.
3. Open a normal HTTPS website **you trust**. Click the VeriFYD Access extension icon to show the overlay.
4. Verify that **Next**, **Select**, **Pointer**, **Auto-scan** and the on-screen keyboard still work (no camera required).
5. Click **Start camera** in the VeriFYD overlay. A separate **VeriFYD Access · Private Eye Gaze Camera** window opens. Click **Start camera** in that trusted extension window and allow camera access there.
6. Keep the camera window *visible beside the website*. Watch the local preview and center your face. In the website's VeriFYD panel select **Calibrate gaze**; follow the five large yellow dots with your eyes while keeping your head mostly steady.
7. Once calibrated, the virtual yellow pointer should move with eye direction in **Pointer mode**. Double-blink to activate an ordinary link/button. Switch to **Controls mode** to single-blink forward and double-blink select.
8. Press **Pause** on the website or **Camera off** in the camera window to immediately stop camera use. Closing the overlay also closes its camera window.

If the camera window reports **MediaPipe assets missing**, repeat Step 3 (model installation) and reload the extension. If it reports camera denied, check Chrome's extension camera permission and any operating-system camera restrictions. Some managed school/work computers may block cameras or unpacked extensions entirely.

## Support and safety boundaries

- **Webpages only:** virtual pointer does **not** move the actual Windows/macOS pointer, control browser chrome, click protected dialogs, or work inside every cross-origin iframe.
- **Calibration is experimental:** eye/head pose, lighting, glasses, camera location and viewing distance can affect accuracy. No medical-grade or hands-free reliability claim is made. Physical webcam tests on your actual hardware are still pending.
- **Browser sites can observe on-page DOM changes** (for example, the visible selection ring/pointer). They are **not** granted direct access to extension camera frames or the derived sample messages.
- **Passwords/payment inputs:** the on-screen keyboard does not type into password or payment-code fields. Automated gesture selection can still be risky on high-stakes sites; use Pause and independent confirmation.
- **Reduced-mobility fallback:** existing switch scanning, F7/F8/F9, speech guidance, on-screen keyboard, and slow pointer stepping remain available if tracking fails.
- **Privacy:** no video storage, network frame uploads, automatic website access, or remote model loading at runtime. The installer needs internet once to download audited third-party model assets.

## Validation

- Node syntax checks for all extension JS modules: passed.
- Gaze geometry, calibration, single/double blink rules: passed mocked unit checks.
- Extension-origin IPC authorization and anti-spoofing checks: passed mocked tests.
- Chromium webpage overlay & Phase A regression smoke test: passed.
- True installed Chromium extension smoke test on this machine: **blocked by sandbox network administrator policy** (`ERR_BLOCKED_BY_ADMINISTRATOR` on a local test site).
- Real camera/Face Landmarker inference and physical Eye Gaze calibration: **not yet tested** (model binary downloads unavailable in build environment).

Read `TEST_REPORT.md` for reproducible tests and `NEXT_PHASE.md` for priorities (physical iPad/Windows verification, calibration tuning, eventual system mouse native companion).
