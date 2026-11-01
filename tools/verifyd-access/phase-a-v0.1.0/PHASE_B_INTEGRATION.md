# Phase B: bring VeriFYD Circle Eye Gaze into VeriFYD Access

## Source-of-truth components in the October 31 Horizons export

- `apps/web/src/hooks/useTrustVoiceMovementDetector.js`
- `apps/web/src/hooks/useTrustVoiceAdaptiveAccess.js`
- `apps/web/src/utils/trustVoiceAdaptiveAccessStorage.js`

These files remain **unchanged**. The browser prototype does not import Horizons' React application or expose its session tokens.

## Input intent contract (already implemented)

The extension's isolated content script owns `globalThis.__verifydAccessControllerV1.dispatchIntent(intent)`; **not** exposed to arbitrary page JavaScript. This is the adapter point for a future local camera or switch source.

Supported intent examples (trusted extension world only):

```js
controller.dispatchIntent({ type: 'NEXT' });
controller.dispatchIntent({ type: 'SELECT' });
controller.dispatchIntent({ type: 'MODE', mode: 'cursor' });
controller.dispatchIntent({ type: 'MOVE', direction: 'left' });
controller.dispatchIntent({ type: 'SCROLL', direction: 'down' });
controller.dispatchIntent({ type: 'PAUSE' });
controller.dispatchIntent({ type: 'RESUME' });
```

Do **not** bridge these intents from a host page via unrestricted `window.postMessage`: any website might forge such a message. Prefer extension runtime channels with verified sender, origin and user consent, or call directly from a camera adapter executing in the isolated extension context.

## Safety and accessibility requirements

- Request camera permission only from a clearly labeled opt-in control on a trusted extension-origin page.
- Process camera frames on-device; don't transmit or retain face images or camera footage.
- Never allow webcam input alone to authorize a purchase, delete an account, send private messages, or accept high-risk browser dialogs without confirmation.
- Use calibration on the device with on-screen guide; provide slow scanning, dwell sensitivity and blink thresholds.
- Use the existing single-blink advance and double-blink select design as a configurable option—not the only accessibility method.
- Provide a switch fallback, a large Pause/Resume button and keyboard shortcuts.
- Support website accessibility APIs (focus, buttons, links, inputs) in preference to artificial OS mouse events.
- Extend later with a native Windows/macOS accessibility companion only after per-device consent and testing.

## Why the Apple keyboard project stays paused

Apple iOS custom keyboard extensions can contribute text input but cannot universally operate the system pointer and arbitrary app interfaces. Safari Web Extensions can reuse much of this web layer after an Xcode conversion and device testing. On compatible iPads, system Eye Tracking can navigate Safari and parts of iPadOS. No iPad build, app-store upload, or Circle publishing is performed by this Phase A ZIP.
