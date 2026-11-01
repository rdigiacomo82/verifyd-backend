# Next: production-ready Eye Gaze, then desktop mouse

1. Physical desktop verification: real MediaPipe models, calibration, iris and blink sensitivity, performance, font scaling, head stability and low-light ergonomics.
2. Improve calibration: adaptive gaze sensitivity, restartable calibration, eye-position bias correction, dwell only with deliberate opt-in, optional long blink as alternate selection.
3. Build a packaged, reviewed extension for Chrome/Edge. Bundle the tested model assets with license notices; remove unpacked developer install instructions before distribution.
4. Safari Web Extension conversion: iPad/iPhone packaging and camera entitlement/permission review. Consider Apple's built-in system Eye Tracking for supported iPads rather than claiming universal browser gaze control.
5. Separate native Windows/macOS companion with explicit accessibility permission to move the **actual OS mouse** outside the browser. Use a clearly labeled, local-only IPC link from the sensor rather than granting a browser extension system-wide control.
6. Android accessibility service proof-of-concept with user authorization, selectable app scope, clear indication when active, and store policy review.
7. For all platforms, audit the risks of accidental clicks, purchases, messages, privacy exposure and other high-impact actions; default to safe confirmation.
