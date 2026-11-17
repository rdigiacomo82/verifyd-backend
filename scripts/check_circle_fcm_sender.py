#!/usr/bin/env python3
"""Read-only Firebase FCM sender preflight for the existing Render secret.

No device token, notification campaign, message delivery, DB write, or key
creation. The FCM request uses validate_only=true. Never print access tokens,
private keys, client email, credential JSON, or HTTP error response bodies.
Run inside the trusted Render Shell, never in an untrusted client.
"""
import json
import os
import re
from pathlib import Path

FCM_SCOPE = "https://www.googleapis.com/auth/firebase.messaging"
PREFLIGHT_TOPIC = "verifyd-circle-preflight"


def request_body():
    return {
        "validate_only": True,
        "message": {
            "topic": PREFLIGHT_TOPIC,
            "data": {"probe": "credential-validation-only"},
        },
    }


def check_metadata(path, project):
    """Return a non-secret failure code or None; never expose file contents."""
    if not path or not Path(path).is_file():
        return "SECRET_FILE_MISSING"
    try:
        with open(path, encoding="utf-8") as stream:
            metadata = json.load(stream)
    except (OSError, UnicodeError, ValueError):
        return "SECRET_FILE_UNREADABLE"
    if not isinstance(metadata, dict) or metadata.get("type") != "service_account":
        return "NOT_A_SERVICE_ACCOUNT"
    if metadata.get("project_id") != project:
        return "PROJECT_ID_MISMATCH"
    if not metadata.get("private_key") or not metadata.get("client_email"):
        return "SERVICE_ACCOUNT_FIELDS_MISSING"
    return None


def token_encryption_ready():
    """Validate existing persistent token-encryption key without showing it."""
    value = (os.getenv("VERIFYD_NATIVE_CALL_TOKEN_ENCRYPTION_KEY") or "").strip()
    if not value:
        return False
    try:
        from cryptography.fernet import Fernet
        Fernet(value.encode())
        return True
    except (ValueError, TypeError):
        return False


def run():
    project = (os.getenv("VERIFYD_ANDROID_FIREBASE_PROJECT_ID") or "").strip()
    path = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{5,62}", project):
        print("FCM_PREFLIGHT RESULT=FAIL REASON=PROJECT_ID_UNSET_OR_INVALID")
        return 1
    failure = check_metadata(path, project)
    if failure:
        print("FCM_PREFLIGHT RESULT=FAIL REASON=" + failure)
        return 1
    print("FCM_PREFLIGHT SECRET_FILE=OK PROJECT_MATCH=OK")
    encryption_ready = token_encryption_ready()
    print("FCM_PREFLIGHT TOKEN_ENCRYPTION=" + ("OK" if encryption_ready else "MISSING_OR_INVALID"))

    try:
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_file(
            path, scopes=[FCM_SCOPE]
        )
        credentials.refresh(Request())
        if not credentials.valid or not credentials.token:
            print("FCM_PREFLIGHT RESULT=FAIL REASON=OAUTH_TOKEN_UNAVAILABLE")
            return 1
    except Exception:
        # Do not log OAuth exception strings or credential information.
        print("FCM_PREFLIGHT RESULT=FAIL REASON=OAUTH_REFRESH_ERROR")
        return 1
    print("FCM_PREFLIGHT OAUTH_REFRESH=OK")

    try:
        import httpx
        response = httpx.post(
            "https://fcm.googleapis.com/v1/projects/" + project + "/messages:send",
            headers={"Authorization": "Bearer " + credentials.token},
            json=request_body(),
            timeout=12,
            follow_redirects=False,
        )
    except Exception:
        print("FCM_PREFLIGHT RESULT=FAIL REASON=FCM_NETWORK_ERROR")
        return 1
    if response.status_code != 200:
        # HTTP status is useful, unlike the potentially sensitive response body.
        print("FCM_PREFLIGHT RESULT=FAIL FCM_HTTP_STATUS=" + str(response.status_code))
        return 1
    if not encryption_ready:
        print("FCM_PREFLIGHT RESULT=PARTIAL FCM_SEND_PERMISSION=OK ENROLLMENT_BLOCKED=TOKEN_ENCRYPTION")
        return 2
    print("FCM_PREFLIGHT RESULT=PASS FCM_SEND_PERMISSION=OK DELIVERY=NONE_VALIDATE_ONLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
