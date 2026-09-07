# VeriFYD Lens MVP Architecture

Recommended first platform: Windows 11 + Edge/Chrome.

## Browser extension
- captures explicit Scan Link action
- later intercepts supported downloads
- sends source URL to local Lens agent
- displays results

## Local Lens agent
- binds to localhost only
- downloads to quarantine
- never executes quarantined files
- SHA-256
- MIME/extension checks
- Windows Defender integration
- calls existing VeriFYD backend for supported files
- release/delete actions

## Browser limitation
A browser cannot universally inspect the full contents of a file before any bytes are downloaded. Lens should cancel/intercept supported downloads and perform the actual download into quarantine through the local agent. Authenticated POST/blob downloads need a scan-after-temporary-download fallback.

## Production hardening
- per-install secret between extension and agent
- signed installer
- code signing
- auto-update
- native messaging
- minimal logging
- explicit cloud-upload policy
