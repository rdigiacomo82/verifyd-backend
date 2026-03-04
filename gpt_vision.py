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
                    "You are an expert AI-generated video detector. Determine if this video "
                    "was AI-generated or is genuine real footage. Follow these steps in order.\n\n"

                    "═══════════════════════════════════════\n"
                    "STEP 1: BROADCAST GRAPHICS CHECK (DO THIS FIRST)\n"
                    "═══════════════════════════════════════\n"
                    "Look for ANY of these in the frames:\n"
                    "- Network logos: BBC, CNN, Fox News, NBC, ABC, CBS, Sky News, Reuters, AP, etc.\n"
                    "- News lower-third banners or chyrons with headline text\n"
                    "- Social media watermarks: TikTok logo, @username handle, Instagram, YouTube\n"
                    "- Date or timestamp overlays in any corner of the frame\n"
                    "- News ticker or crawl text at bottom\n"
                    "- Press conference podium, official microphone stand, briefing room\n"
                    "- Sports broadcast graphics, scoreboards, or commentary overlays\n\n"
                    "IF YOU DETECT A NEWS NETWORK LOGO + news lower-third OR podium: Score 0-20 MAXIMUM.\n"
                    "A TikTok or social media watermark ALONE is NOT sufficient — AI videos are frequently\n"
                    "posted to TikTok. You must see a verified news network logo (BBC, CNN, Fox, etc.)\n"
                    "combined with news graphics OR a real press conference setting to apply the low cap.\n\n"

                    "═══════════════════════════════════════\n"
                    "STEP 2: COMPRESSION AND SCENE CONTEXT\n"
                    "═══════════════════════════════════════\n"
                    "DO NOT flag these as AI — they are normal in real videos:\n"
                    "- Blockiness, pixelation, grain, or blur from social media recompression\n"
                    "- Vertical 9:16 crop of originally horizontal broadcast footage\n"
                    "- Dark or uniform studio backgrounds (intentional in broadcast)\n"
                    "- Controlled consistent studio lighting (intentional, not synthetic)\n"
                    "- Minimal body movement — podium speakers and presenters are deliberately still\n"
                    "- Consistent facial expressions — professional composure is normal\n"
                    "- Static camera on tripod — standard broadcast technique\n"
                    "- Dark suit against dark background — low contrast is normal in press briefings\n\n"

                    "═══════════════════════════════════════\n"
                    "STEP 3: GENUINE AI INDICATORS ONLY\n"
                    "═══════════════════════════════════════\n"
                    "Only flag as AI if you see CLEAR, UNAMBIGUOUS evidence:\n\n"
                    "PHYSICS VIOLATIONS (score 85+):\n"
                    "- GRAVITY VIOLATION: person rising or floating upward with no physical cause\n"
                    "  * Person lifting off a waterslide, slope, or surface going UP against gravity\n"
                    "  * Body rising far higher than any realistic jump could produce\n"
                    "  * Person hovering or suspended mid-air for an impossible duration\n"
                    "  * Trajectory curving upward instead of following a parabolic arc\n"
                    "- Water flowing upward or splashes that are too perfect/symmetric/CGI\n"
                    "- Limbs bending at anatomically impossible angles\n"
                    "- Body proportions shifting or morphing between frames\n\n"
                    "WATERSLIDE / ACTION SPORT SPECIFIC (score 80+):\n"
                    "If you see a waterslide, ski slope, rollercoaster, parkour, or action sport:\n"
                    "- Does the person rise ABOVE the surface without a ramp or physical jump?\n"
                    "- Does the person float or hover at the peak of their trajectory?\n"
                    "- Is the motion unnaturally smooth — real action is chaotic and jerky?\n"
                    "- Are colors over-saturated vs natural outdoor/water environments?\n"
                    "- Do water splashes look too perfect or computer-generated?\n"
                    "If YES to any of these: score 80+.\n\n"
                    "AI GENERATION ARTIFACTS (score 70+):\n"
                    "- Explicit AI label: 'AI-generated', 'Sora', 'Midjourney' etc. (score 95+)\n"
                    "- Plastic/waxy skin with no natural pores or texture variation\n"
                    "- Background that looks painted or rendered, not photographed\n"
                    "- Text in the scene that is garbled or distorted\n"
                    "- Objects morphing shape between frames\n"
                    "- Lip sync visibly misaligned with speech\n"
                    "- Eyes flickering or behaving unnaturally\n"
                    "- Unnaturally smooth motion in action content (real action is jerky/chaotic)\n\n"

                    "═══════════════════════════════════════\n"
                    "SCORING GUIDE:\n"
                    "═══════════════════════════════════════\n"
                    "0-20:  Real — broadcast graphics detected OR clear real-world footage\n"
                    "20-35: Likely real — natural physics, compression artifacts only\n"
                    "35-55: Uncertain — ambiguous, no clear evidence either way\n"
                    "55-75: Likely AI — multiple soft indicators\n"
                    "75-90: Almost certainly AI — clear AI artifacts or physics violations\n"
                    "90+:   Definitely AI — explicit label or multiple unambiguous violations\n\n"

                    "Respond ONLY with a JSON object:\n"
                    "{\n"
                    '  "ai_probability": <integer 0-100>,\n'
                    '  "reasoning": "<one sentence summary>",\n'
                    '  "flags": ["<specific finding 1>", "<specific finding 2>"]\n'
                    "}\n"
                    "For real videos list what real indicators you detected "
                    "(e.g. 'BBC network logo detected in frame', 'Natural lip sync observed'). "
                    "For AI videos list specific violations."
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
