"""Unauthenticated, zero-user-data verification of guarded voicemail service readiness."""
import json
import urllib.error
import urllib.request

BASE = "https://verifyd-backend.onrender.com/trust-voice"

def get(path):
    request = urllib.request.Request(BASE + path, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8") or "{}")

status, health = get("/voicemail/health")
assert status == 200, f"Voicemail health failed with HTTP {status}"
assert health.get("feature") == "voicemail_intelligence", health
print("Voicemail health code:", status)
print("Feature enabled:", bool(health.get("enabled")))
print("Transcription provider configured on web:", bool(health.get("transcription_configured")))
print("Public business ingestion:", bool(health.get("public_business_ingestion")))
assert health.get("public_business_ingestion") is False

code, _ = get("/voicemail/inbox-state")
print("Unauthenticated inbox-state HTTP:", code)
assert code in {401, 404}, "Anonymous voicemail inbox data must never be accessible"
if health.get("enabled"):
    assert code == 401, "Enabled inbox must enforce authentication"
else:
    assert code == 404, "Disabled inbox should not expose feature"

code, _ = get("/voicemail/fake_attachment_id/review")
print("Unauthenticated voicemail review HTTP:", code)
assert code in {401, 404}, "No anonymous voicemail analysis allowed"
print("PASS — guarded voicemail endpoints and private recipient access")
