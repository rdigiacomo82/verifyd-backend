# ============================================================
#  VeriFYD — gpt_vision.py
#
#  GPT-4o vision analysis for AI video detection.
#  Extracts key frames from video and sends to GPT-4o for
#  semantic analysis — detects impossible elements, AI art
#  artifacts, unnatural physics, and content anomalies that
#  pure signal analysis cannot catch.
#
#  Returns a 0-100 AI confidence score + reasoning text.
#  Designed to run alongside detector.py for a combined verdict.
# ============================================================

import os
import cv2
import base64
import logging
import tempfile
import numpy as np
from typing import Optional

log = logging.getLogger("verifyd.gpt_vision")

# ─────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GPT_MODEL      = "gpt-4o"
MAX_FRAMES     = 8      # frames to send to GPT-4o (more = better detection)
FRAME_QUALITY  = 70     # JPEG quality for base64 encoding (lower = cheaper)
MAX_DIMENSION  = 512    # resize frames to this max dimension (cost control)


# ─────────────────────────────────────────────────────────────
#  Frame extraction
# ─────────────────────────────────────────────────────────────
def extract_key_frames(video_path: str, n_frames: int = MAX_FRAMES) -> list:
    """
    Extract n evenly-spaced frames from the video.
    Returns list of base64-encoded JPEG strings.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("gpt_vision: cannot open %s", video_path)
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return []

    # Pick evenly spaced frame indices, skip first 10% and last 20%
    start = max(0, int(total_frames * 0.10))
    end   = min(total_frames - 1, int(total_frames * 0.80))
    indices = [int(start + (end - start) * i / (n_frames - 1)) for i in range(n_frames)]

    frames_b64 = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize to max dimension for cost control
        h, w = frame.shape[:2]
        if max(h, w) > MAX_DIMENSION:
            scale = MAX_DIMENSION / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # Encode as JPEG base64
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, FRAME_QUALITY])
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        frames_b64.append(b64)

    cap.release()
    log.info("gpt_vision: extracted %d frames from %s", len(frames_b64), video_path)
    return frames_b64


# ─────────────────────────────────────────────────────────────
#  GPT-4o Analysis
# ─────────────────────────────────────────────────────────────
def analyze_frames_with_gpt(frames_b64: list) -> dict:
    """
    Send frames to GPT-4o for semantic AI detection analysis.
    Returns dict with: ai_probability (0-100), reasoning, flags
    """
    if not OPENAI_API_KEY:
        log.warning("gpt_vision: OPENAI_API_KEY not set — skipping GPT analysis")
        return {"ai_probability": 50, "reasoning": "GPT analysis unavailable", "flags": []}

    if not frames_b64:
        return {"ai_probability": 50, "reasoning": "No frames extracted", "flags": []}

    try:
        import urllib.request
        import urllib.error
        import json

        # Build message content with all frames
        content = [
            {
                "type": "text",
                "text": (
                    "You are an expert AI-generated and VFX video detector. Analyze these video frames "
                    "and determine if this video was generated or enhanced by AI or VFX.\n\n"
                    "CRITICAL: Check for ANY text in the frames that says 'AI-generated', "
                    "'AI enhanced', 'AI created', 'made with AI', or similar labels. "
                    "If you see such text, score ai_probability at 95+.\n\n"
                    "Look for these STRONG AI/VFX indicators:\n"
                    "- Text overlay stating 'AI-generated', 'AI-enhanced', or similar\n"
                    "- Impossible or physically impossible scenes (ocean waves indoors, "
                    "wave pools on ship decks, impossible weather, supernatural events)\n"
                    "- VFX compositing — elements that look digitally added to real footage\n"
                    "- Water, fire, or weather that looks CGI or unnaturally perfect\n"
                    "- Obvious AI art style (plastic skin, unnaturally perfect faces, dreamlike quality)\n"
                    "- Objects morphing or changing shape between frames\n"
                    "- Distorted or nonsensical text and signs\n"
                    "- Background that looks painted or rendered, not photographed\n"
                    "- Creatures or beings that cannot exist in reality\n"
                    "- Movie set with blue/green screen visible in background\n"
                    "- Viral 'unexplained phenomena' style content with impossible subjects\n\n"
                    "These are NOT reliable AI indicators — do NOT penalize for:\n"
                    "- Low resolution or compression artifacts (common in phone videos)\n"
                    "- Slight blur or noise (normal for real cameras)\n"
                    "- Text overlays or captions (these are edits, not AI generation)\n\n"
                    "Real video indicators:\n"
                    "- Natural camera motion and shake with no impossible elements\n"
                    "- Consistent lighting and physics throughout\n"
                    "- Ordinary everyday scenes that obey physics\n\n"
                    "Respond ONLY with a JSON object in this exact format:\n"
                    "{\n"
                    '  "ai_probability": <integer 0-100>,\n'
                    '  "reasoning": "<one sentence summary>",\n'
                    '  "flags": ["<specific anomaly 1>", "<specific anomaly 2>"]\n'
                    "}\n"
                    "Where ai_probability=100 means definitely AI/VFX, 0 means definitely real.\n"
                    "For ordinary phone videos with no AI artifacts, score 15-30.\n"
                    "For videos with impossible composited elements or impossible physics, score 75-95.\n"
                    "For videos explicitly labeled AI-generated, score 95+.\n"
                    "flags should list specific anomalies detected (empty array if none)."
                )
            }
        ]

        # Add each frame as an image
        for b64 in frames_b64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high"  # high detail needed to read text overlays
                }
            })

        payload = {
            "model": GPT_MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 300,
            "temperature": 0.1,   # low temperature for consistent scoring
        }

        import time
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            },
            method="POST"
        )

        # Retry up to 3 times on 429 rate limit
        data = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 2:
                    wait = (attempt + 1) * 5  # 5s, 10s
                    log.warning("GPT rate limited, retrying in %ds (attempt %d)", wait, attempt+1)
                    time.sleep(wait)
                    continue
                raise
        if data is None:
            raise RuntimeError("GPT API failed after 3 attempts")

        raw_text = data["choices"][0]["message"]["content"].strip()
        log.info("gpt_vision raw response: %s", raw_text[:200])

        # Strip markdown code fences if present
        if "```" in raw_text:
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        result = json.loads(raw_text.strip())

        # Validate and clamp
        ai_prob = max(0, min(100, int(result.get("ai_probability", 50))))
        reasoning = str(result.get("reasoning", ""))[:500]
        flags = [str(f)[:100] for f in result.get("flags", [])][:10]

        log.info("gpt_vision: ai_probability=%d  flags=%s", ai_prob, flags)
        return {"ai_probability": ai_prob, "reasoning": reasoning, "flags": flags}

    except Exception as e:
        log.error("gpt_vision error: %s", e)
        return {"ai_probability": 50, "reasoning": f"GPT analysis error: {str(e)[:100]}", "flags": []}


# ─────────────────────────────────────────────────────────────
#  Public interface
# ─────────────────────────────────────────────────────────────
def gpt_vision_score(video_path: str) -> dict:
    """
    Main entry point. Extract frames and run GPT-4o analysis.
    Returns dict with ai_probability (0-100), reasoning, flags.
    """
    if not OPENAI_API_KEY:
        log.warning("gpt_vision: OPENAI_API_KEY not configured")
        return {"ai_probability": 50, "reasoning": "GPT vision not configured", "flags": [], "available": False}

    frames = extract_key_frames(video_path)
    if not frames:
        return {"ai_probability": 50, "reasoning": "Could not extract frames", "flags": [], "available": False}

    result = analyze_frames_with_gpt(frames)
    result["available"] = True
    return result
