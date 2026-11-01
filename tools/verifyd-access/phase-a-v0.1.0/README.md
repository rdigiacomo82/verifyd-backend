# VeriFYD Access — Phase A, v0.1.0 (standalone prototype)

A **reusable browser-level accessibility controller**. This is deliberately **not** a replacement of VeriFYD Circle or its Xcode keyboard extension; the former stays unchanged, and the Apple keyboard Track A remains paused.

## Supported today

- Chromium browsers on Windows/macOS/Linux: Chrome and Edge (MV3 unpacked extension)
- Start or stop on an explicitly selected website with the extension icon, or Ctrl+Shift+9 (Mac: Command+Shift+9)
- Controls mode: F7 highlights the next visible button/link/input; F8 activates it; F9 starts/stops slow automatic switch scanning (5 seconds by default; adjustable)
- Pointer mode: directional buttons or keyboard arrow keys move a **virtual webpage pointer** in 25-pixel increments by default (adjustable); F8 clicks the visible page element under it
- Scroll up/down, an explicit pause/resume, and an **optional** 3.2-second dwell selection (OFF by default)
- Optional slow spoken guidance; adjustable scan speed and pointer step
- Accessible on-screen QWERTY keyboard for basic text fields; native input-event integration supports many React-based websites
- All functionality runs locally in the currently authorized browser tab
- No external network calls, no analytics, no storage, and no access to other tabs

## Not implemented yet

**Camera-based Eye Gaze, blink detection, OS-level mouse movement, Android device-wide taps, and iOS/iPadOS Safari packaging are NOT included in v0.1.0.** The existing VeriFYD Eye Gaze/Movement detector remains in Circle. Next phase will adapt its calibration, blink thresholds, and movement events to the input-intent contract here after physical-device validation. An extension cannot control browser chrome, protected dialogs, other applications, or every cross-origin embedded iframe.

## Quick start (Chrome or Microsoft Edge on desktop)

1. Unzip this project folder.
2. Open `chrome://extensions` (Chrome) or `edge://extensions` (Edge).
3. Enable **Developer mode**.
4. Select **Load unpacked** and choose this project's **extension** folder (containing `manifest.json`).
5. Open a normal HTTPS website, such as an ordinary information page with links and buttons.
6. Click the **VeriFYD Access** extension icon to show the accessible control panel.
7. Choose **Next** to highlight a control, then **Select** to activate it. Try F7/F8/F9 or switch to **Pointer** mode.
8. Click the extension icon again to close and remove VeriFYD Access from that tab.

**Security:** install only on a device you control; this is a development prototype, not a tested assistive medical device or production browser extension. The extension is opt-in per tab, has only `activeTab` and `scripting`, does not read credentials or send data, and has no camera permission. Dwell may click unintended controls if explicitly enabled—keep it off on financial, medical, or other high-stakes sites. Pause before leaving the device.

## How this ties into the former Apple keyboard build

The earlier `VeriFYDAdaptiveKeyboard` Track A remains paused. Do not remove or overwrite the Xcode project. The `navigation-core.js` functions are a small, platform-independent starting point. Phase B will connect gaze/blink/switch events from the existing VeriFYD adaptive detection pipeline to this core as *intents*, rather than tying those events to a particular webpage or social network.

## Next engineering milestones

1. Migrate/refactor the existing `useTrustVoiceMovementDetector.js` into a sensor adapter: gaze direction, blink event, sensitivity, calibration, enable/disable. **Never** feed raw video or credentials to websites.
2. Add secure camera lifecycle in a privileged extension page, with clear opt-in, pause, and fallbacks; bundle any ML models locally (Chrome MV3 forbids remotely hosted executable code).
3. Move the overlay into a browser tab permission model that users can grant for selected sites after testing; improve iframe and SPA handling.
4. Convert the extension to an Apple Safari Web Extension project for iPad/iPhone with Xcode packaging; leverage system Eye Tracking rather than claiming cross-app mouse control from webpage JavaScript.
5. Add a separately installed, consent-gated native accessibility companion for actual OS-wide pointer control on Windows/macOS; investigate Android accessibility-service prototype.

## Repository layout

- `extension/manifest.json` — explicit permissions and entry points
- `extension/background.js` — click/shortcut on-demand injection
- `extension/navigation-core.js` — pure pointer/scan utilities
- `extension/content.js` — website controls, navigation and keyboard
- `extension/content.css` — isolated prefixed styles
- `tests/` — unit tests and browser smoke tests

This project deliberately does **not** alter Hostinger Horizons, Render, the current Circle app, the PWA, or the prior keyboard Xcode project.
