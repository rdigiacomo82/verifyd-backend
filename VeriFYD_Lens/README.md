# VeriFYD Lens
## Don't Download Blind.

VeriFYD Lens is a desktop + browser protection layer that checks a download before it is released to the user's normal Downloads folder.

MVP flow:
Browser -> Lens -> Quarantine -> Scan -> Result -> Release or Delete

Initial scan layers:
- URL/source checks
- file type / extension mismatch checks
- SHA-256 fingerprint
- Windows Defender hook
- VeriFYD cloud analysis hook
- unified Security / Authenticity / Trust result

Important: AI-generated content is not automatically malicious. Lens separates security risk from authenticity/AI signals.
