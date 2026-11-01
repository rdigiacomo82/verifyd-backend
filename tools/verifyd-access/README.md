# VeriFYD Access — standalone accessibility browser extension

These are **versioned source snapshots** of the VeriFYD Access prototype, kept separate from the active VeriFYD Circle / Render backend.

- [Phase A v0.1.0](./phase-a-v0.1.0/README.md) — Chrome/Edge webpage navigation, virtual pointer, scanning and accessible keyboard.
- [Phase B v0.2.0](./phase-b-v0.2.0/README.md) — Phase A plus opt-in camera-based Eye Gaze adapter and calibration.

**Install only Phase B** for current testing. Run `phase-b-v0.2.0/scripts/Install-MediaPipe-Assets.ps1` locally to install the external MediaPipe assets before camera testing, then load `phase-b-v0.2.0/extension` using Chrome/Edge **Load unpacked**.

**Limitations:** Phase B camera model assets are not checked into this repository, and physical webcam / real Eye Gaze validation is incomplete. The extension controls the authorized webpage, not the actual operating-system pointer or other applications.

**Deployment:** This source branch is for version control only. **No Render service is required**, and this project is not integrated into or deployed through the VeriFYD Circle backend.

Original project previews are available in the ZIP archives shared in the ChatGPT conversation; the screenshots are omitted from this source snapshot to keep the repository code-only.
