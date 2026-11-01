# VeriFYD Access v0.1.0 — validation report

## Passed

- JavaScript syntax: `background.js`, `navigation-core.js`, `content.js` — `node --check`
- `manifest.json` — parsed as JSON; Chrome MV3 with `activeTab` and `scripting` only
- Navigation core unit test: circular scan indexing, pointer movement, coordinate bounds, label cleanup
- Chromium automated webpage smoke test: insert overlay; scan a link and page button; click page button; select a text field; type and backspace; avoid typing into a password field; type into email field; activate checkbox; verify adapter pause/resume; select Pointer; move pointer; pause; close; re-inject and toggle without duplicates
- No code initiates a network request or reads browser cookies, persisted tokens, or external tabs
- Browser screenshot available as `PREVIEW.png`

## Not yet validated

- Installing/unpacking through Chrome/Edge's actual extension toolbar on a real user desktop (Chromium's extension service worker could not be exercised in this container's headless/Xvfb test environment)
- Behavior on a wide sample of external websites, including shadow DOM, cross-origin iframes and protected pages
- Physical switch, gaze camera, blink input, iPad Safari and Android device-wide use
- An OS-level mouse movement or controls outside the webpage (not implemented)

## Important usage notes

Dwell is off by default. The prototype's on-screen keyboard will not operate password or payment/autofill code inputs. Keep dwell off on sensitive sites. This prototype is not a certified assistive device.
