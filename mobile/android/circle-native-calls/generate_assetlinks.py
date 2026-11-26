#!/usr/bin/env python3
"""Generate production Digital Asset Links for the *signed* VeriFYD Circle app.

Use the Play App Signing SHA-256 app-signing certificate fingerprint, NOT a
developer/debug/upload key. This script never deploys the result to vfvid.com.
"""
import argparse
import json
import re

parser = argparse.ArgumentParser()
parser.add_argument("--play-app-signing-sha256", required=True,
                    help="SHA-256 from Play Console > App integrity > App signing")
args = parser.parse_args()
fingerprint = args.play_app_signing_sha256.strip().upper()
if not re.fullmatch(r"(?:[0-9A-F]{2}:){31}[0-9A-F]{2}", fingerprint):
    parser.error("Expected exactly 32 uppercase/lowercase hex octets separated by ':'")
if fingerprint in {"00:" * 31 + "00", "FF:" * 31 + "FF"}:
    parser.error("Placeholder or invalid signing fingerprint")
print(json.dumps([{
    "relation": [
        "delegate_permission/common.handle_all_urls",
        "delegate_permission/common.use_as_origin",
    ],
    "target": {
        "namespace": "android_app",
        "package_name": "com.verifyd.circle",
        "sha256_cert_fingerprints": [fingerprint],
    },
}], indent=2))
