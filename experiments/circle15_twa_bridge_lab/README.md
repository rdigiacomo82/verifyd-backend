# VeriFYD Circle 15 — isolated Android TWA PostMessage lab

**This is a standalone test page, not the VeriFYD Circle website or its backend.**

Published via a separate Render **static site** `verifyd-circle15-bridge-lab`, URL:
`https://verifyd-circle15-bridge-lab.onrender.com/voice/`.

- Site serves `/.well-known/assetlinks.json` for **debug** package `com.verifyd.circle.pilot.dev` and the **debug signing certificate SHA-256**, not the production key.
- No logins, cookies, analytics, backend calls, sensitive user information, microphone access, or registration.
- A secure static page accepts the native-origin/port hello and can request **local** native status only. This is **not** proof of session binding or call delivery.
- Website source is in `experiments/circle15_twa_bridge_lab/site` only.
- Render auto-deploy is **OFF** to avoid unreviewed changes to the lab.
- Android pilot needs a **debug-only** change in `CircleTwaBridgeActivity`: ORIGIN `https://verifyd-circle15-bridge-lab.onrender.com`; PAGE `https://verifyd-circle15-bridge-lab.onrender.com/voice/`; add a nativepilotDebug resource override of `assetStatements` **retaining vfvid.com and adding this lab origin**. Do not modify the release TWA or its signing identity.

## No changes permitted

Leave `https://vfvid.com/.well-known/assetlinks.json` as `[]`, leave production `main` Android launcher and Horizons unchanged, and keep backend bridge PR #108 unmerged and flag OFF.

## Validation

Check:
`https://verifyd-circle15-bridge-lab.onrender.com/.well-known/assetlinks.json`
responds 200, `application/json`, contains only the pilot debug package and test key; `/voice/` returns the static test harness.

Render's deploy may report live before public HTTPS verification succeeds. Verify from a browser/Windows before testing on device. A browser alone will correctly display *Waiting* until Chrome TWA native PostMessage is connected.

**This branch is a standalone lab. Never merge into the production backend without independent review.**