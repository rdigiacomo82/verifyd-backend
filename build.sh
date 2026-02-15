#!/usr/bin/env bash
set -e

echo "=== VeriFYD Build ==="

# System deps
apt-get update -qq
apt-get install -y -qq ffmpeg

# Python deps
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Build Complete ==="
