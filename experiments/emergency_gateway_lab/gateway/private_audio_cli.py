"""Run a single local synthetic SIP/PCMU-RTP test without a web server.

Run as: python -m gateway.private_audio_cli
Network restrictions are enforced by private_sip_rtp.py (127.0.0.1 only).
"""
import asyncio
import json
import sys
from .private_sip_rtp import run_private_audio_test


def main() -> int:
    result = asyncio.run(run_private_audio_test())
    print(json.dumps(result, indent=2))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())