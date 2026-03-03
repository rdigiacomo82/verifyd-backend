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
#
#  v3: Added threading semaphore to prevent concurrent GPT
#  bursts from triggering 429 rate limit errors. Max 3
#  simultaneous GPT calls regardless of user concurrency.
#  Increased retry waits from 5s/10s to 15s/30s/45s.
# ============================================================

import os
import cv2
import base64
import logging
import tempfile
import threading
import numpy as np
from typing import Optional

log = logging.getLogger("verifyd.gpt_vision")

# ─────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GPT_MODEL      = "gpt-4o"
MAX_FRAMES     = 10
FRAME_QUALITY  = 75
MAX_DIMENSION  = 768

# ─────────────────────────────────────────────────────────────
#  Concurrency limiter
#  Max 3 simultaneous GPT calls — prevents burst 429 errors
#  when multiple users submit videos at the same time.
# ─────────────────────────────────────────────────────────────
_gpt_semaphore = threading.Semaphore(3)


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

    start   = max(0, int(total_frames * 0.05))
    end     = min(total_frames - 1, int(total_frames * 0.95))
    indices = [int(start + (end - start) * i / (n_frames - 1)) for i in range(n_frames)]

    frames_b64 = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        if max(h, w) > MAX_DIMENSION:
            scale = MAX_DIMENSION / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

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
        import time

        content = [
            {
                "type": "text",
                "text": (
                    "You are an expert AI-generated video detector with deep knowledge of physics "
                    "and human biomechanics. Analyze these video frames carefully.\n\n"

                    "═══════════════════════════════════════\n"
                    "PHYSICS VIOLATIONS — HIGHEST PRIORITY\n"
                    "═══════════════════════════════════════\n"
                    "These are the strongest indicators of AI generation. If you see ANY of these, "
                    "score ai_probability at 85 or higher:\n\n"

                    "GRAVITY VIOLATIONS:\n"
                    "- A person lifting off or floating above a surface (waterslide, ground, etc.) "
                    "without a visible jump, ramp, or external force\n"
                    "- Body hovering or rising upward against gravity in a context where this is "
                    "physically impossible (e.g. rising off a waterslide mid-slide)\n"
                    "- Objects or people suspended in mid-air longer than physics allows\n"
                    "- A person's trajectory defying the parabolic arc that gravity produces\n\n"

                    "WATER PHYSICS VIOLATIONS:\n"
                    "- Water flowing upward or sideways against gravity\n"
                    "- Splashes that look CGI — too perfect, too symmetric, or unrealistic\n"
                    "- Water disappearing or appearing instantaneously\n"
                    "- Wave or splash size inconsistent with the force that caused it\n\n"

                    "BODY PHYSICS VIOLATIONS:\n"
                    "- Limbs bending at anatomically impossible angles\n"
                    "- Body proportions shifting or morphing between frames\n"
                    "- Clothing or hair behaving differently from what physics predicts\n"
                    "- Movement that is too smooth, too fast, or too perfect for a human\n\n"

                    "ENVIRONMENTAL PHYSICS:\n"
                    "- Any element in the scene behaving in a way that common sense says\n"
                    "  cannot happen in that real-world context\n"
                    "- Scene elements that appear digitally composited onto real footage\n\n"

                    "═══════════════════════════════════════\n"
                    "OTHER AI/VFX INDICATORS — STRONG SIGNALS\n"
                    "═══════════════════════════════════════\n"
                    "- Text overlay stating 'AI-generated', 'AI-enhanced', or similar (score 95+)\n"
                    "- Obvious AI art style: plastic/waxy skin, unnaturally perfect faces\n"
                    "- Background that looks painted or rendered rather than photographed\n"
                    "- Objects morphing or changing shape between frames\n"
                    "- Distorted or nonsensical text and signs\n"
                    "- Creatures or beings that cannot exist in reality\n"
                    "- VFX compositing with visible seams or inconsistent lighting\n\n"

                    "═══════════════════════════════════════\n"
                    "DO NOT penalize for these — they are normal in real videos:\n"
                    "═══════════════════════════════════════\n"
                    "- Low resolution or compression artifacts (common in phone videos)\n"
                    "- Slight blur, noise, or shaky camera (normal for real cameras)\n"
                    "- Text overlays or captions added in post-production\n"
                    "- Slow motion effects (normal video technique)\n"
                    "- Natural athletic or acrobatic movement that is fast or impressive\n\n"

                    "═══════════════════════════════════════\n"
                    "SCORING GUIDE:\n"
                    "═══════════════════════════════════════\n"
                    "0-20:  Definitely real — natural physics, no AI artifacts\n"
                    "20-40: Likely real — minor anomalies, could be compression\n"
                    "40-60: Uncertain — some suspicious elements but not conclusive\n"
                    "60-80: Likely AI — multiple anomalies or one clear physics violation\n"
                    "80-95: Almost certainly AI — clear physics violations or AI artifacts\n"
                    "95+:   Definitely AI — explicitly labeled or multiple obvious violations\n\n"

                    "Respond ONLY with a JSON object in this exact format:\n"
                    "{\n"
                    '  "ai_probability": <integer 0-100>,\n'
                    '  "reasoning": "<one sentence summary of the key finding>",\n'
                    '  "flags": ["<specific anomaly 1>", "<specific anomaly 2>"]\n'
                    "}\n"
                    "flags should list specific anomalies detected (empty array if none).\n"
                    "Be specific in flags — e.g. 'Person lifts off waterslide at frame 3 "
                    "with no jump or ramp visible' rather than just 'unnatural movement'."
                )
            }
        ]

        for b64 in frames_b64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high"
                }
            })

        payload = {
            "model":       GPT_MODEL,
            "messages":    [{"role": "user", "content": content}],
            "max_tokens":  400,
            "temperature": 0.1,
        }

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            },
            method="POST"
        )

        # ── Semaphore: max 3 concurrent GPT calls ─────────────
        # Queues excess requests instead of bursting all at once,
        # preventing 429 errors under high user concurrency.
        data = None
        with _gpt_semaphore:
            # Retry up to 4 times with increasing waits: 15s, 30s, 45s
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429 and attempt < 3:
                        wait = (attempt + 1) * 15   # 15s, 30s, 45s
                        log.warning("GPT rate limited, retrying in %ds (attempt %d)",
                                    wait, attempt + 1)
                        time.sleep(wait)
                        continue
                    raise

        if data is None:
            raise RuntimeError("GPT API failed after 4 attempts")

        raw_text = data["choices"][0]["message"]["content"].strip()
        log.info("gpt_vision raw response: %s", raw_text[:200])

        if "```" in raw_text:
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        result    = json.loads(raw_text.strip())
        ai_prob   = max(0, min(100, int(result.get("ai_probability", 50))))
        reasoning = str(result.get("reasoning", ""))[:500]
        flags     = [str(f)[:100] for f in result.get("flags", [])][:10]

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