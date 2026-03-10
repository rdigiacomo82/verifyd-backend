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

                    "CINEMATIC / ANIMAL / WILDLIFE CONTENT (score 70+):\n"
                    "If you see animals, wildlife, nature scenes, or cinematic footage,\n"
                    "apply ALL of these checks — AI animal videos are a common deepfake category:\n\n"
                    "- FUR / FEATHER TEXTURE (most reliable tell):\n"
                    "  Look closely at the animal's coat, fur, or feathers.\n"
                    "  REAL: Individual hair strands visible, directional growth patterns,\n"
                    "  natural clumping, matting, moisture variation, dirt/debris in fur.\n"
                    "  AI: Fur appears as a smooth uniform mass — no individual strands,\n"
                    "  looks like velvet, plush toy, or CGI render. Too perfect, too clean.\n"
                    "  For dark-furred animals (gorilla, bear, black cat): AI renders dark fur\n"
                    "  as a flat dark mass with no micro-detail. Real dark fur still shows\n"
                    "  individual hair texture and light catching on strand tips.\n"
                    "  → If fur looks like a render or plush toy: score 75+\n\n"
                    "- SKIN / FACE TEXTURE:\n"
                    "  REAL: Animal facial skin (around eyes, muzzle, pads) is dry, wrinkled,\n"
                    "  leathery, pored — visibly aged and textured.\n"
                    "  AI: Skin appears as a smooth gradient — no pores, no wrinkles, no\n"
                    "  surface irregularity. Looks like polished rubber or painted silicone.\n"
                    "  → If skin looks unnaturally smooth for the species: score 70+\n\n"
                    "- EYES (very reliable — AI eyes are almost always wrong):\n"
                    "  REAL animal eyes: small, irregular catchlights, wet-surface imperfect\n"
                    "  reflections, mostly iris/pupil visible, natural occlusion from lids.\n"
                    "  AI animal eyes: large symmetric perfectly round catchlights, too much\n"
                    "  white sclera visible (human-like), eyes too bright and 'glassy',\n"
                    "  reflections that look like perfectly placed studio lights.\n"
                    "  → If eyes look too perfect, too bright, or too human: score 75+\n\n"
                    "- BACKGROUND (AI compositing tell):\n"
                    "  REAL outdoor footage: background has natural depth variation, real\n"
                    "  leaves have irregular edges, lighting varies across the scene.\n"
                    "  AI background: unnaturally uniform synthetic bokeh (flat blur),\n"
                    "  vegetation looks repetitive/symmetric (rendered leaves), background\n"
                    "  looks like a stock photo or nature documentary backdrop.\n"
                    "  Subject and background often have mismatched lighting direction.\n"
                    "  → If background looks rendered or like a stock photo: score 70+\n\n"
                    "- MOVEMENT QUALITY:\n"
                    "  REAL animals move with weight, muscle tension, and physical impact.\n"
                    "  Knuckle-walking gorilla: impact shudder, muscle definition visible.\n"
                    "  Running animals: fur ripples, body bounces with inertia.\n"
                    "  AI animals: movement too smooth and fluid, like CGI animation.\n"
                    "  No physical weight or inertia — moves like it is floating.\n"
                    "  → If movement looks animated rather than physical: score 70+\n\n"
                    "- GRAIN / SENSOR NOISE:\n"
                    "  REAL camera footage: luminance noise visible in dark areas (fur, shadows).\n"
                    "  AI renders: completely noise-free, or noise is too uniform/regular.\n"
                    "  Dark fur in real footage should have visible grain. AI dark fur is clean.\n"
                    "  → If dark regions are completely noise-free: score 65+\n\n"
                    "- LIGHTING CONSISTENCY:\n"
                    "  REAL outdoor: dappled shadows, uneven highlights from tree cover,\n"
                    "  natural rim lighting, subsurface scattering in ears/thin skin areas.\n"
                    "  AI: lighting follows a perfect gradient, looks like a render farm output,\n"
                    "  too diffused and perfect, no harsh or unexpected shadows.\n"
                    "  → If lighting looks like a professional CGI render: score 70+\n\n"
                    "COMBINED ANIMAL SCORING: If you observe 3+ of the above tells, score 80+.\n"
                    "If you observe fur texture issues AND eye issues together, score 80+.\n\n"


                    "CROWD / EMERGENCY / REACTION CONTENT (score 75+):\n"
                    "If you see crowds, emergencies, accidents, police, or public reactions:\n"
                    "- REACTION TIMING: Do bystanders react with appropriate urgency and surprise?\n"
                    "  Real emergencies trigger immediate, instinctive human reactions — flinching,\n"
                    "  running, screaming, chaos. AI crowds often react too slowly, too calmly,\n"
                    "  or not at all relative to the severity of what is happening.\n"
                    "- CROWD BEHAVIOR: Are crowd members moving independently and chaotically?\n"
                    "  Real crowds have organic, unpredictable individual movement — people bumping,\n"
                    "  turning, pointing, using phones. AI crowds often move in unnaturally uniform\n"
                    "  or synchronized patterns, or people seem frozen/static in the background.\n"
                    "- POLICE/OFFICIAL RESPONSE: Do officers or officials respond with appropriate\n"
                    "  urgency? Real police respond fast to active threats. AI scenes often show\n"
                    "  officers standing or moving slowly even during active emergencies.\n"
                    "- AUDIO-VISUAL SYNC (if inferrable from visual cues): Do mouths match speech?\n"
                    "  Do crowd sounds match crowd movement? Do impact sounds match what you see?\n"
                    "  AI videos frequently have audio that is mismatched, too clean, or generic\n"
                    "  stock-sound-style reactions that don't match the specific visual event.\n"
                    "- EMOTIONAL AUTHENTICITY: Do facial expressions match the situation?\n"
                    "  Real emergency footage captures raw, uncontrolled human emotion.\n"
                    "  AI faces often show neutral or slightly wrong expressions for the context.\n"
                    "- SCENE LOGIC: Does everyone's behavior make sense for what is happening?\n"
                    "  In real emergencies, people instinctively seek safety, help others, or record.\n"
                    "  AI scenes often have people acting as if the event isn't happening.\n"
                    "If reactions are too calm/slow, crowd movement is synchronized, or official\n"
                    "response is disproportionately delayed: score 75+.\n\n"

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
                    "0-10:  Clearly real — broadcast verified OR obvious real-world footage\n"
                    "       with natural motion, sensor noise, camera shake, and no AI tells.\n"
                    "10-25: Likely real — natural physics and lighting, only compression\n"
                    "       artifacts, slight ambiguity but no AI indicators present.\n"
                    "25-45: Uncertain — some ambiguity, no strong evidence either way.\n"
                    "45-65: Likely AI — multiple soft indicators present.\n"
                    "65-85: Almost certainly AI — clear AI artifacts or physics violations.\n"
                    "85+:   Definitely AI — explicit AI label or multiple unambiguous violations.\n\n"
                    "IMPORTANT: If you see NO AI indicators at all — natural motion, real\n"
                    "camera grain/shake, plausible physics, compression artifacts — score 0-15.\n"
                    "Do NOT score 30-40 just because you are uncertain. Uncertainty with\n"
                    "no AI evidence = low score (0-20). Only score above 25 if you see\n"
                    "actual AI indicators, not just the absence of proof it is real.\n\n"

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
    # NEW v6 context fields
    quad_cov     = ctx.get("quad_cov")         # low = uniform AI render focus
    fg_bg_ratio  = ctx.get("fg_bg_ratio")      # high = unnatural depth
    motion_sync  = ctx.get("motion_sync")      # low = lockstep AI crowd
    # Behavioral signals
    flow_entropy = ctx.get("flow_dir_entropy") # low = uniform AI crowd motion
    peak_ratio   = ctx.get("peak_to_mean_ratio") # low = no reaction spikes
    # v9 content type
    content_type = ctx.get("content_type", "cinematic")  # selfie / talking_head / action / cinematic
    is_talking_head = content_type == "talking_head"
    is_selfie = content_type == "selfie"

    lines.append("═══════════════════════════════════════")
    lines.append("SIGNAL DETECTOR PRE-ANALYSIS (v4 — measured before you see these frames):")
    lines.append("Use this data to guide and confirm your visual analysis.\n")

    # ── Content type notice ──
    if is_talking_head:
        lines.append("📱 CONTENT TYPE: TALKING-HEAD / ACTIVE PORTRAIT")
        lines.append("   Signal detector identified this as a real person talking, walking, or moving on camera.")
        lines.append("   IMPORTANT: The following are NORMAL for this content type — do NOT treat as AI signals:")
        lines.append("   • Low saturation variance — skin tones + neutral clothing naturally dominate")
        lines.append("   • Consistent edge density — single subject in a scene, not a crowd")
        lines.append("   • Uniform motion sync — one person moves as one unit, not a scripted crowd")
        lines.append("   • Uniform sharpness across frame — phone cameras focus on one subject")
        lines.append("   • Smooth steady motion — a person walking or talking moves smoothly by nature")
        lines.append("   • No sudden motion spikes — walking/talking has no emergency reaction bursts")
        lines.append("   For this content type, ONLY flag AI if you see: unnatural skin smoothness,")
        lines.append("   missing pores/imperfections, AI-typical hair (too perfect or floating),")
        lines.append("   inconsistent lighting across frames, or morphing/glitching artifacts.")
        lines.append("   A realistic-looking person on a phone video is almost certainly REAL.\n")
    elif is_selfie:
        lines.append("📱 CONTENT TYPE: SELFIE / STATIC PORTRAIT")
        lines.append("   Signal detector identified this as a phone selfie or portrait video.\n")

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
        if sat_std < 5.0 and not is_talking_head and not is_selfie:
            lines.append(f"⚠ Frozen lighting detected (sat_std={sat_std:.2f}) — "
                         f"natural lighting always varies. Possible AI render. "
                         f"Look for unnaturally perfect, studio-quality lighting on the subject.")
        elif sat_std < 5.0 and (is_talking_head or is_selfie):
            lines.append(f"✓ Low sat variance (sat_std={sat_std:.2f}) — expected for portrait with skin/neutral tones.")
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

    # ── v6: Render uniformity signals ──
    if quad_cov is not None:
        if quad_cov < 0.40:
            lines.append(f"⚠ UNIFORM RENDER FOCUS (quad_cov={quad_cov:.3f}) — "
                         f"sharpness is identical across all frame quadrants. "
                         f"Real cameras have natural depth-of-field variation. "
                         f"This is a strong AI render signature. Look for unnaturally "
                         f"sharp subjects against suspiciously blurred/rendered backgrounds.")
        elif quad_cov < 0.50 and not is_talking_head:
            lines.append(f"⚠ Low depth-of-field variation (quad_cov={quad_cov:.3f}) — "
                         f"focus is more uniform than expected for real camera footage.")

    if fg_bg_ratio is not None and fg_bg_ratio > 900:
        lines.append(f"⚠ EXTREME SUBJECT/BACKGROUND CONTRAST (fg_bg={fg_bg_ratio:.0f}) — "
                     f"the subject is rendered at extreme sharpness vs the background. "
                     f"This unnatural depth ratio is a strong AI compositing signature.")

    if motion_sync is not None and motion_sync < 0.09 and not is_talking_head and not is_selfie:
        lines.append(f"⚠ LOCKSTEP CROWD MOTION (sync={motion_sync:.3f}) — "
                     f"left and right halves of frame move in near-identical patterns. "
                     f"Real crowds have independent chaotic movement. AI crowds are scripted.")

    lines.append("\n═══════════════════════════════════════")
    lines.append("Now examine the frames with this context in mind.")
    if bg_drift is not None and bg_drift < 3.0:
        lines.append("→ Pay close attention to whether the background looks rendered or artificial.")
    if sat_std is not None and sat_std < 5.0 and not is_talking_head and not is_selfie:
        lines.append("→ Check if the subject's fur, skin, or texture looks unnaturally smooth or plastic.")
        lines.append("→ For animals: look closely at fur for individual hair strands (real) vs smooth mass (AI).")
        lines.append("→ Check animal eyes — AI eyes have large symmetric catchlights and look too bright.")
    if is_talking_head:
        lines.append("→ PORTRAIT FOCUS: Examine skin texture for natural pores, imperfections, subtle redness.")
        lines.append("→ Check hair for individual strands, fly-aways, and natural lighting on hair tips.")
        lines.append("→ Look for natural micro-expressions and spontaneous blinks/eye movement.")
        lines.append("→ Check teeth, ears, and hands if visible — these are hard for AI to render naturally.")
    if flicker_std is not None and flicker_std > 4.0:
        lines.append("→ Look for inconsistencies in fine details between frames.")
    if quad_cov is not None and quad_cov < 0.50 and not is_talking_head:
        lines.append("→ RENDER FLAG: Focus is suspiciously uniform. Check if this looks like a CGI render.")
        lines.append("  Look for the 'uncanny valley' quality — too perfect to be real camera footage.")
    if flow_entropy is not None and flow_entropy < 1.5:
        lines.append("→ BEHAVIORAL FLAG: Motion analysis detected unnaturally uniform crowd/scene movement.")
        lines.append("  Look carefully at whether bystanders react with appropriate urgency and chaos.")
        lines.append("  Real emergencies produce unpredictable individual movement — AI scenes do not.")
    if peak_ratio is not None and peak_ratio < 3.0 and not is_talking_head:
        lines.append("→ BEHAVIORAL FLAG: No dramatic motion spikes detected — real emergency footage")
        lines.append("  always has sudden reaction bursts. This scene's motion is too smooth/gradual.")
    lines.append("")

    return "\n".join(lines)


