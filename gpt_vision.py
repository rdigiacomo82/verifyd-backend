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
#  v3: Added threading semaphore, increased retry waits.
#  v4: Added cinematic/animal/nature AI detection prompt section.
#      GPT now explicitly checks for:
#      - Plastic/waxy fur and skin texture
#      - Unnaturally smooth animal movement
#      - CGI-quality lighting and reflections
#      - Background that looks rendered vs photographed
#      - Oversaturated "cinematic AI" color grading
#      Updated physics context builder to pass sat_std,
#      bg_drift, and flicker_std signals to GPT.
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
GPT_MODEL      = os.environ.get("VERIFYD_GPT_MODEL", "gpt-4o")
MAX_FRAMES     = 6
FRAME_QUALITY  = 80
MAX_DIMENSION  = 512

# ─────────────────────────────────────────────────────────────
#  Concurrency limiter — max 3 simultaneous GPT calls
# ─────────────────────────────────────────────────────────────
_gpt_semaphore = threading.Semaphore(3)


# ─────────────────────────────────────────────────────────────
#  Frame extraction
# ─────────────────────────────────────────────────────────────
def extract_key_frames(video_path: str, n_frames: int = MAX_FRAMES) -> list:
    """
    Extract n evenly-spaced frames from the video.
    Returns list of base64-encoded JPEG strings.
    Converts WebM/MKV to MP4 first since cv2 cannot read WebM reliably.
    """
    import subprocess, shutil

    converted_path = None
    ext = os.path.splitext(video_path)[1].lower()
    if ext in ('.webm', '.mkv', '.ogg'):
        try:
            tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            converted_path = tmp.name
            tmp.close()
            result = subprocess.run(
                ['ffmpeg', '-y', '-i', video_path, '-c:v', 'libx264',
                 '-preset', 'ultrafast', '-crf', '28', '-an', converted_path],
                capture_output=True, timeout=60
            )
            if result.returncode == 0 and os.path.exists(converted_path):
                log.info("gpt_vision: converted %s -> mp4 for frame extraction", ext)
                video_path = converted_path
            else:
                log.warning("gpt_vision: ffmpeg conversion failed, trying original")
                if os.path.exists(converted_path):
                    os.remove(converted_path)
                converted_path = None
        except Exception as e:
            log.warning("gpt_vision: conversion error: %s — trying original", e)
            converted_path = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("gpt_vision: cannot open %s", video_path)
        if converted_path and os.path.exists(converted_path):
            os.remove(converted_path)
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
    if converted_path and os.path.exists(converted_path):
        os.remove(converted_path)
    log.info("gpt_vision: extracted %d frames from %s", len(frames_b64), video_path)
    return frames_b64


# ─────────────────────────────────────────────────────────────
#  GPT-4o Analysis
# ─────────────────────────────────────────────────────────────
def analyze_frames_with_gpt(frames_b64: list, physics_summary: str = "") -> dict:
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
                    (physics_summary + "\n\n") if physics_summary else ""
                ) + (
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
                    "IMPORTANT: A TikTok or Instagram watermark ALONE is NOT sufficient to lower the score.\n"
                    "AI-generated videos are frequently posted to TikTok. Social media watermarks prove\n"
                    "distribution, NOT authenticity. Only apply the low-score cap when you see a verified\n"
                    "NEWS NETWORK LOGO (BBC, CNN, Fox, Reuters, AP etc.) combined with news graphics\n"
                    "OR a real press conference/briefing room setting.\n\n"

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

                    "CINEMATIC / ANIMAL / NATURE CONTENT (score 70+):\n"
                    "If you see animals, wildlife, nature scenes, or cinematic footage:\n"
                    "- TEXTURE: Does fur, skin, or feathers look waxy, plastic, or too smooth?\n"
                    "  Real animal fur has individual hair strands, dirt, matting, natural variation.\n"
                    "  AI fur looks uniformly perfect, like a plush toy or CGI render.\n"
                    "- MOVEMENT: Is animal movement unnaturally fluid or graceful?\n"
                    "  Real animals move with weight, effort, and imperfection.\n"
                    "  AI animals move like animations — too smooth, too perfectly coordinated.\n"
                    "- EYES: Do animal eyes look glassy, perfectly reflective, or unnaturally bright?\n"
                    "  Real animal eyes are imperfect with natural occlusion and variation.\n"
                    "- LIGHTING: Does lighting look like a professional CGI render — perfectly\n"
                    "  diffused, no harsh shadows, no natural imperfections like lens flare or\n"
                    "  dappled light? Real outdoor footage has uncontrolled, imperfect lighting.\n"
                    "- BACKGROUND: Does the background look painted, rendered, or like a stock\n"
                    "  photo backdrop rather than a real place with depth and imperfection?\n"
                    "- COLORS: Are colors over-saturated or color-graded in a way that looks\n"
                    "  like a movie poster rather than natural footage?\n"
                    "- SCENE COHERENCE: Do the subject and background match in terms of lighting\n"
                    "  angle, time of day, and shadow direction? AI often composites these poorly.\n"
                    "If you see plastic fur, CGI lighting, or rendered backgrounds: score 70+.\n\n"

                    "AI GENERATION ARTIFACTS (score 70+):\n"
                    "- Explicit AI label: 'AI-generated', 'Sora', 'Midjourney' etc. (score 95+)\n"
                    "- Plastic/waxy skin with no natural pores or texture variation\n"
                    "- Background that looks painted or rendered, not photographed\n"
                    "- Text in the scene that is garbled or distorted\n"
                    "- Objects morphing shape between frames\n"
                    "- Lip sync visibly misaligned with speech\n"
                    "- Eyes flickering or behaving unnaturally\n"
                    "- Unnaturally smooth motion in action or animal content\n\n"

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
                    "detail": "low"
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

        data = None
        with _gpt_semaphore:
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429 and attempt < 3:
                        wait = (attempt + 1) * 15
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


def gpt_vision_score_with_context(frames_b64: list, physics_context: dict) -> dict:
    """
    Run GPT-4o analysis with physics engine context pre-loaded.
    physics_context dict contains signal detector findings to guide GPT.
    """
    if not OPENAI_API_KEY:
        return {"ai_probability": 50, "reasoning": "GPT vision not configured", "flags": [], "available": False}

    if not frames_b64:
        return {"ai_probability": 50, "reasoning": "Could not extract frames", "flags": [], "available": False}

    physics_summary = _build_physics_summary(physics_context)
    result = analyze_frames_with_gpt(frames_b64, physics_summary)
    result["available"] = True
    return result


def _build_physics_summary(ctx: dict) -> str:
    """Convert physics context dict into a natural language summary for GPT."""
    if not ctx:
        return ""

    lines = []
    signal_score = ctx.get("signal_score")
    vert_flow    = ctx.get("vert_flow")
    upward_ratio = ctx.get("upward_ratio")
    accel_std    = ctx.get("accel_std")
    low_corr     = ctx.get("low_corr_count")
    saturation   = ctx.get("avg_saturation")
    sharpness    = ctx.get("avg_sharpness")
    # NEW v4 context fields
    sat_std      = ctx.get("sat_frame_std")
    bg_drift     = ctx.get("bg_drift")
    flicker_std  = ctx.get("flicker_std")

    lines.append("═══════════════════════════════════════")
    lines.append("SIGNAL DETECTOR PRE-ANALYSIS (v4 — measured before you see these frames):")
    lines.append("Use this data to guide and confirm your visual analysis.\n")

    if signal_score is not None:
        label = ("HIGH — strong AI indicators" if signal_score > 60
                 else "LOW — consistent with real video" if signal_score < 40
                 else "MODERATE — ambiguous")
        lines.append(f"Overall AI signal score: {signal_score}/100 ({label})")

    # ── Physics / motion ──
    if vert_flow is not None:
        if vert_flow < -1.0:
            lines.append(f"⚠ GRAVITY VIOLATION: Mean vertical flow = {vert_flow:.2f} "
                         f"(negative = upward motion against gravity). "
                         f"Look for the person rising or floating above the surface.")
        elif vert_flow < -0.3:
            lines.append(f"⚠ Upward motion tendency: vertical flow = {vert_flow:.2f}. "
                         f"Check if person appears to defy gravity.")
        else:
            lines.append(f"✓ Gravity-consistent motion: vertical flow = {vert_flow:.2f}.")

    if upward_ratio is not None:
        pct = upward_ratio * 100
        if pct > 25:
            lines.append(f"⚠ {pct:.0f}% of frames show strong upward motion — "
                         f"physically impossible for slide/action content.")
        elif pct < 10:
            lines.append(f"✓ Only {pct:.0f}% upward-motion frames — consistent with real physics.")

    if accel_std is not None:
        if accel_std < 1.5:
            lines.append(f"⚠ Unnaturally smooth trajectory (accel_std={accel_std:.2f}). "
                         f"Real action is chaotic. AI motion is smooth.")
        else:
            lines.append(f"✓ Natural chaotic trajectory (accel_std={accel_std:.2f}).")

    if low_corr is not None and low_corr > 5:
        lines.append(f"⚠ {low_corr} frame discontinuities — content jumps typical of AI generation.")

    # ── Color / lighting ──
    if saturation is not None and saturation > 100:
        lines.append(f"⚠ Over-saturated colors (saturation={saturation:.0f}) — "
                     f"real outdoor footage is typically less saturated.")

    if sat_std is not None:
        if sat_std < 5.0:
            lines.append(f"⚠ Frozen lighting detected (sat_std={sat_std:.2f}) — "
                         f"natural lighting always varies. Possible AI render. "
                         f"Look for unnaturally perfect, studio-quality lighting on the subject.")
        elif sat_std > 22.0:
            lines.append(f"⚠ Unstable color/lighting (sat_std={sat_std:.2f}) — "
                         f"flickering or inconsistent saturation typical of AI generation artifacts.")
        else:
            lines.append(f"✓ Natural lighting variation (sat_std={sat_std:.2f}).")

    # ── Background stability ──
    if bg_drift is not None:
        if bg_drift < 3.0:
            lines.append(f"⚠ Frozen background (bg_drift={bg_drift:.2f}) — "
                         f"background corners are nearly static, suggesting a rendered/AI scene. "
                         f"Look for a background that appears painted or computer-generated.")
        elif bg_drift > 25.0:
            lines.append(f"⚠ Warping background (bg_drift={bg_drift:.2f}) — "
                         f"background is shifting unnaturally between frames, a common AI artifact. "
                         f"Look for background elements that change or morph.")
        else:
            lines.append(f"✓ Natural background movement (bg_drift={bg_drift:.2f}).")

    # ── Temporal consistency ──
    if flicker_std is not None:
        if flicker_std > 4.0:
            lines.append(f"⚠ High temporal inconsistency (flicker_std={flicker_std:.2f}) — "
                         f"frames are inconsistently rendered, typical of AI video generation. "
                         f"Look for subtle frame-to-frame changes in texture, lighting, or detail.")
        elif flicker_std < 1.0:
            lines.append(f"✓ Temporally consistent frames (flicker_std={flicker_std:.2f}).")

    if sharpness is not None and sharpness < 150:
        lines.append(f"⚠ Low sharpness ({sharpness:.0f}) — AI videos tend to be softer than real camera footage.")

    lines.append("\n═══════════════════════════════════════")
    lines.append("Now examine the frames with this context in mind.")
    if bg_drift is not None and bg_drift < 3.0:
        lines.append("→ Pay close attention to whether the background looks rendered or artificial.")
    if sat_std is not None and sat_std < 5.0:
        lines.append("→ Check if the subject's fur, skin, or texture looks unnaturally smooth or plastic.")
    if flicker_std is not None and flicker_std > 4.0:
        lines.append("→ Look for inconsistencies in fine details between frames.")
    lines.append("")

    return "\n".join(lines)
