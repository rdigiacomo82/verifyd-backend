# ============================================================
#  VeriFYD — gpt_vision.py  v5
#
#  GPT-4o vision analysis for AI video detection.
#
#  v5 ARCHITECTURE — Structured 12-Dimension Rubric:
#
#  Previously: one narrative prompt → one number (inconsistent,
#  undebuggable, can't tune individual dimensions).
#
#  Now: GPT scores 12 independent dimensions 0-10 with a
#  required reason per dimension. Python computes the final
#  score using content-type-aware weights. This gives:
#    - Reproducible scores (same video → same result)
#    - Per-dimension debugging (know exactly what fired)
#    - Generator fingerprinting (Sora vs Kling vs Pika)
#    - Tunable weights per content type without re-prompting
#    - Full evidence trail stored per job
#
#  DIMENSIONS (0=definitely real, 10=definitely AI):
#    1.  skin_texture       — pores/grain vs waxy/smooth
#    2.  hair_detail        — individual strands vs helmet mass
#    3.  eye_quality        — natural imperfection vs glassy/symmetric
#    4.  motion_physics     — inertia/weight vs floaty/smooth CGI
#    5.  background_realism — photographed depth vs rendered/stock
#    6.  lighting_coherence — consistent real source vs AI-perfect
#    7.  temporal_stability — frame consistency vs flicker/morph
#    8.  color_naturalism   — camera palette vs hyperreal/flat
#    9.  crowd_behavior     — organic chaos vs synchronized/scripted
#    10. text_objects       — readable/stable vs garbled/morphing
#    11. physics_violations — plausible motion vs impossible motion
#    12. generator_artifacts— no AI evidence vs explicit AI label
#
#  FRAME STRATEGY:
#    - 8 frames at 'high' detail for person content (skin/eye texture)
#    - 8 frames at 'low' detail for non-person content
#    - Spread: 5%–95% of video to avoid title/end cards
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
MAX_FRAMES     = 8           # increased from 6
FRAME_QUALITY  = 85          # increased from 80
MAX_DIMENSION  = 768         # increased from 512

# ─────────────────────────────────────────────────────────────
#  Concurrency limiter — max 3 simultaneous GPT calls
# ─────────────────────────────────────────────────────────────
_gpt_semaphore = threading.Semaphore(3)


# ─────────────────────────────────────────────────────────────
#  Dimension definitions & content-type-aware weights
# ─────────────────────────────────────────────────────────────

DIMENSIONS = [
    "skin_texture",
    "hair_detail",
    "eye_quality",
    "motion_physics",
    "background_realism",
    "lighting_coherence",
    "temporal_stability",
    "color_naturalism",
    "crowd_behavior",
    "text_objects",
    "physics_violations",
    "generator_artifacts",
]

# Base weights (unnormalized — normalized at runtime)
_BASE_WEIGHTS = {
    "skin_texture":       1.5,
    "hair_detail":        1.2,
    "eye_quality":        1.3,
    "motion_physics":     1.5,
    "background_realism": 1.0,
    "lighting_coherence": 0.8,
    "temporal_stability": 1.2,
    "color_naturalism":   0.8,
    "crowd_behavior":     0.8,
    "text_objects":       0.6,
    "physics_violations": 2.0,   # always high — impossible physics = certain AI
    "generator_artifacts":2.5,   # always highest — explicit label = certain AI
}

# Per-content-type multipliers applied on top of base weights
_CONTENT_WEIGHTS = {
    "talking_head": {
        "skin_texture": 2.5, "hair_detail": 2.0, "eye_quality": 2.5,
        "motion_physics": 0.7, "crowd_behavior": 0.2, "background_realism": 0.6,
    },
    "selfie": {
        "skin_texture": 2.5, "hair_detail": 2.0, "eye_quality": 2.5,
        "motion_physics": 0.4, "crowd_behavior": 0.2, "background_realism": 0.6,
    },
    "single_subject": {
        "skin_texture": 2.0, "hair_detail": 1.5, "eye_quality": 1.5,
        "background_realism": 1.5, "motion_physics": 1.2,
    },
    "action": {
        "motion_physics": 2.0, "physics_violations": 2.5,
        "temporal_stability": 1.5, "crowd_behavior": 1.2,
        "color_naturalism": 1.8,   # boosted: hypersat is a key AI tell in action videos
        "background_realism": 1.4, # AI action has rendered backgrounds
        "skin_texture": 1.0,       # raised from 0.7: person action videos need skin check
        "hair_detail": 0.7,
    },
    "cinematic": {
        "background_realism": 2.0, "lighting_coherence": 1.8,
        "color_naturalism": 1.5, "motion_physics": 1.8,
        "skin_texture": 1.2, "hair_detail": 0.7,
    },
    "static": {
        "background_realism": 1.8, "lighting_coherence": 1.8,
        "color_naturalism": 1.5, "temporal_stability": 0.5,
        "motion_physics": 0.4,
    },
}


def _get_weights(content_type: str) -> dict:
    """Return normalized per-dimension weights for a given content type."""
    weights = dict(_BASE_WEIGHTS)
    for dim, mult in _CONTENT_WEIGHTS.get(content_type, {}).items():
        weights[dim] = weights.get(dim, 1.0) * mult
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def _scores_to_ai_probability(scores: dict, content_type: str) -> int:
    """
    Convert per-dimension 0–10 scores → 0–100 AI probability.
    Hard floor rules for unambiguous signals:
      generator_artifacts >= 8 → floor 90
      physics_violations  >= 8 → floor 80
      any dimension       >= 9 → floor 75
    """
    weights = _get_weights(content_type)
    weighted_sum = sum(scores.get(d, 5) * weights.get(d, 0) for d in DIMENSIONS)
    prob = int(round(weighted_sum * 10))
    prob = max(0, min(100, prob))

    if scores.get("generator_artifacts", 0) >= 8:
        prob = max(prob, 90)
    if scores.get("physics_violations", 0) >= 8:
        prob = max(prob, 80)
    if any(scores.get(d, 0) >= 9 for d in DIMENSIONS):
        prob = max(prob, 75)

    return prob


# ─────────────────────────────────────────────────────────────
#  Frame extraction
# ─────────────────────────────────────────────────────────────
def extract_key_frames(video_path: str, n_frames: int = MAX_FRAMES) -> list:
    """
    Extract n evenly-spaced frames from the video.
    Returns list of base64-encoded JPEG strings.
    Converts WebM/MKV/MOV to MP4 first for cv2 reliability.
    """
    import subprocess

    converted_path = None
    ext = os.path.splitext(video_path)[1].lower()
    if ext in ('.webm', '.mkv', '.ogg', '.mov'):
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
                log.info("gpt_vision: converted %s → mp4 for frame extraction", ext)
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
#  The 12-dimension scoring prompt
# ─────────────────────────────────────────────────────────────

_DIMENSION_GUIDE = """\
You are a forensic AI video detection expert. Score each of the 12 dimensions below
from 0 to 10, where:
  0  = strongly real  (clear evidence this is genuine camera footage)
  5  = uncertain / not visible / not applicable to this content
  10 = strongly AI    (clear evidence of synthesis or manipulation)

Score ONLY what you can actually observe. If a dimension is not visible in these
frames (e.g. no people visible → skin_texture is N/A → score 5), score it exactly 5.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 FAST-PATH: BROADCAST / NEWS FOOTAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If you see a NEWS NETWORK LOGO (BBC, CNN, Fox, Reuters, AP, Sky, NBC, ABC, CBS)
COMBINED WITH news lower-thirds, chyrons, or a press briefing room podium:
→ Score ALL 12 dimensions as 1 and explain in reasoning. This is real broadcast.
A TikTok or Instagram watermark ALONE does not qualify — AI videos are posted there.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DO NOT PENALIZE COMPRESSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Blockiness, grain, blur, pixelation from social media re-compression are normal in
REAL videos. Score dimensions on the underlying content — not on compression quality.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 THE 12 DIMENSIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. SKIN_TEXTURE
   0-2 = Real: visible pores, subtle redness/variation, natural color heterogeneity,
         fine lines, natural imperfections, subsurface scattering in ears/nose.
   8-10= AI:   porcelain-smooth, uniform color, airbrushed/plastic quality, no pores,
         no texture variation under any lighting. "Too perfect" is the tell.
   5   = No skin visible in frames, or too compressed to assess.

2. HAIR_DETAIL
   0-2 = Real: individual strands visible, flyaways, directional variation, matte
         roots vs. glossy tips, color variation strand-to-strand.
   8-10= AI:   rendered as a uniform mass, helmet-like, too smooth, all strands
         perfectly aligned, lacks micro-chaos of real hair.
   5   = No hair visible, or too compressed to assess.

3. EYE_QUALITY
   0-2 = Real: irregular catchlights (environmental), natural sclera amount, asymmetric
         iris, realistic moisture, natural lid droop, slight redness.
   8-10= AI:   perfectly symmetric, too bright, circular studio catchlights, too much
         white sclera (wide stare), glassy doll quality, identical in every frame.
   5   = Eyes not visible or too small to assess.

4. MOTION_PHYSICS
   0-2 = Real: subjects decelerate with inertia, hair/clothing lag behind body,
         foot-ground contact causes compression, impact creates visible reaction.
   8-10= AI:   fluid and frictionless, no inertia delay, objects move with perfect
         smoothness like CGI, no secondary motion in hair or fabric.
   5   = Static video or motion not assessable.

5. BACKGROUND_REALISM
   0-2 = Real: natural depth variation, real leaves with irregular edges, real
         bokeh following depth of field, environmental context (dirt, debris, shadows).
   8-10= AI:   synthetic circular bokeh, vegetation looks repetitive/rendered,
         background resembles stock photo, subject and background have mismatched lighting.
   5   = Background not visible or not enough context.

6. LIGHTING_COHERENCE
   0-2 = Real: shadows from consistent real source, rim light follows environment,
         winter outdoor = cool blue-tinted diffuse, indoor = warm uneven.
   8-10= AI:   too diffuse and perfect (render-farm quality), OR incoherent (shadows
         in different directions), OR subject lit differently from background.
   5   = Lighting not assessable.

7. TEMPORAL_STABILITY
   0-2 = Stable: texture, geometry, and lighting are consistent across frames.
   8-10= Unstable: subtle flickering in fine details, texture changes between frames,
         face geometry drifts slightly, background elements appear/disappear.
   Compare multiple frames carefully. Does any fine detail shift in ways a real camera
   wouldn't produce?

8. COLOR_NATURALISM
   0-2 = Real: slightly muted, natural gamut, color variation across scene, subtle
         color noise in shadows. Outdoor overcast = slightly cool and desaturated.
   8-10= AI:   oversaturated "candy" colors, suspiciously perfect color gradients,
         or bizarrely flat/uniform color with no natural variation.
   5   = Color not assessable.

9. CROWD_BEHAVIOR  (score 5 if no crowd / multiple people in scene)
   0-2 = Real: individuals move independently, unpredictable, genuine emotional
         reactions (flinching, running, recording), organic chaos.
   8-10= AI:   synchronized movement, uniform crowd behavior, bystanders too calm
         for the event severity, people act as if central event isn't happening.

10. TEXT_OBJECTS  (score 5 if no text visible in scene)
    0-2 = Real: visible text (signs, labels, screens) is readable and stable across frames.
    8-10= AI:   text is garbled, morphing, misspelled, or transforms between frames.
          AI generators still struggle with consistent text rendering.

11. PHYSICS_VIOLATIONS  ← MOST IMPORTANT DIMENSION
    0-2 = Real: objects obey gravity, water flows down, human trajectories follow
          parabolic arcs, no body part bends at impossible angles.
    8-10= AI:   people floating upward against gravity on slides/slopes, water flowing
          the wrong direction, limbs bending impossibly, body rising without physical cause.
    Score 10 for any clear, unambiguous physics violation. This is dispositive.

12. GENERATOR_ARTIFACTS  ← SECOND MOST IMPORTANT DIMENSION
    0-2 = No AI labels, watermarks, or known generator signatures visible.
    8-10= Explicit AI evidence: text saying "AI-generated", "Sora", "Kling", "Runway",
          "Pika", "Midjourney", "DALL-E", or visible AI tool watermark.
    6-8 = Strong known generator signature without explicit label:
          Sora: characteristic smooth particle systems, slightly dreamlike depth.
          Kling: portrait-mode compression blur, subtle face warping on motion.
          Pika: color bloom on edges, slightly over-sharpened subjects.
          Runway: cinematic color grading, smooth camera moves, slight temporal flicker.
          HeyGen/D-ID: talking-head with slightly too-smooth skin and perfect eye contact.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Respond ONLY with this exact JSON — no markdown, no preamble, no extra text:
{
  "scores": {
    "skin_texture": <0-10>,
    "hair_detail": <0-10>,
    "eye_quality": <0-10>,
    "motion_physics": <0-10>,
    "background_realism": <0-10>,
    "lighting_coherence": <0-10>,
    "temporal_stability": <0-10>,
    "color_naturalism": <0-10>,
    "crowd_behavior": <0-10>,
    "text_objects": <0-10>,
    "physics_violations": <0-10>,
    "generator_artifacts": <0-10>
  },
  "reasoning": "<one concise sentence overall assessment>",
  "top_flags": ["<most significant finding>", "<second finding>", "<third finding>"],
  "generator_guess": "<Sora | Kling | Runway | Pika | HeyGen | Unknown-AI | Real>"
}
"""


# ─────────────────────────────────────────────────────────────
#  GPT-4o Analysis
# ─────────────────────────────────────────────────────────────
def analyze_frames_with_gpt(frames_b64: list, physics_summary: str = "",
                             content_type: str = "cinematic") -> dict:
    """
    Send frames to GPT-4o for 12-dimension rubric scoring.
    Returns dict: ai_probability, scores, reasoning, flags, generator_guess.
    """
    if not OPENAI_API_KEY:
        log.warning("gpt_vision: OPENAI_API_KEY not set — skipping GPT analysis")
        return _unavailable_result("GPT analysis unavailable — no API key")

    if not frames_b64:
        return _unavailable_result("No frames extracted")

    try:
        import urllib.request
        import urllib.error
        import json
        import time

        # Higher detail for person content where texture matters most.
        # Also use high detail for action content — we need to see skin/color
        # quality in child/person action videos (AI_Child: action but person visible).
        # Low detail only for truly non-person content (cinematic/animal/static).
        img_detail = "low" if content_type in ("cinematic", "static") else "high"

        content = [
            {
                "type": "text",
                "text": (
                    (physics_summary + "\n\n") if physics_summary else ""
                ) + _DIMENSION_GUIDE
            }
        ]

        for b64 in frames_b64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": img_detail
                }
            })

        payload = {
            "model":       GPT_MODEL,
            "messages":    [{"role": "user", "content": content}],
            "max_tokens":  800,         # increased from 400 — rubric needs room
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
        log.info("gpt_vision raw response: %s", raw_text[:300])

        # Strip markdown fences if present
        if "```" in raw_text:
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        result = json.loads(raw_text.strip())

        # Extract and clamp scores
        raw_scores = result.get("scores", {})
        scores = {}
        for dim in DIMENSIONS:
            val = raw_scores.get(dim, 5)
            scores[dim] = max(0, min(10, int(round(float(val)))))

        # Compute weighted AI probability in Python
        ai_prob = _scores_to_ai_probability(scores, content_type)

        reasoning       = str(result.get("reasoning", ""))[:500]
        top_flags       = [str(f)[:150] for f in result.get("top_flags", [])][:5]
        generator_guess = str(result.get("generator_guess", "Unknown"))[:50]

        log.info(
            "gpt_vision: ai_prob=%d  content=%s  generator=%s  scores=%s",
            ai_prob, content_type, generator_guess,
            " ".join(f"{k[:4]}={v}" for k, v in scores.items())
        )

        return {
            "ai_probability":   ai_prob,
            "reasoning":        reasoning,
            "flags":            top_flags,
            "scores":           scores,
            "generator_guess":  generator_guess,
            "available":        True,
        }

    except Exception as e:
        log.error("gpt_vision error: %s", e)
        return {
            "ai_probability":  50,
            "reasoning":       f"GPT analysis error: {str(e)[:100]}",
            "flags":           [],
            "scores":          {},
            "generator_guess": "Unknown",
            "available":       False,
        }


def _unavailable_result(reason: str) -> dict:
    return {
        "ai_probability":  50,
        "reasoning":       reason,
        "flags":           [],
        "scores":          {},
        "generator_guess": "Unknown",
        "available":       False,
    }


# ─────────────────────────────────────────────────────────────
#  Signal detector context → GPT pre-amble
# ─────────────────────────────────────────────────────────────
def _build_physics_summary(ctx: dict) -> str:
    """
    Convert signal detector context dict into a focused visual inspection
    guide for GPT. Only passes signals that have a visual correlate GPT
    can actually confirm — avoids flooding GPT with numbers it can't see.
    """
    if not ctx:
        return ""

    content_type   = ctx.get("content_type", "cinematic")
    signal_score   = ctx.get("signal_score")
    sat_std        = ctx.get("sat_frame_std")
    bg_drift       = ctx.get("bg_drift")
    flicker_std    = ctx.get("flicker_std")
    quad_cov       = ctx.get("quad_cov")
    motion_sync    = ctx.get("motion_sync")
    avg_saturation = ctx.get("avg_saturation")
    flow_entropy   = ctx.get("flow_dir_entropy")
    vert_flow      = ctx.get("vert_flow")

    is_person = content_type in ("talking_head", "selfie", "single_subject")
    is_action = content_type == "action"

    lines = ["═══ SIGNAL DETECTOR PRE-ANALYSIS ═══"]
    lines.append("Pixel-level detector measured these signals before you see the frames.")
    lines.append("Use them to guide your visual inspection — confirm or override with what you observe.\n")

    ct_labels = {
        "talking_head":   "📱 PERSON VIDEO — talking head or active portrait",
        "selfie":         "📱 SELFIE — static portrait",
        "single_subject": "🎥 SINGLE PERSON IN FRAME — landscape",
        "action":         "⚡ ACTION / SPORT CONTENT",
        "cinematic":      "🎬 CINEMATIC / NATURE / ANIMAL",
        "static":         "📷 STATIC / LOW-MOTION CONTENT",
    }
    lines.append(ct_labels.get(content_type, f"Content type: {content_type}"))

    if signal_score is not None:
        tier = ("HIGH — strong AI indicators" if signal_score > 60
                else "LOW — consistent with real footage" if signal_score < 35
                else "MODERATE — ambiguous")
        lines.append(f"Signal score: {signal_score}/100 ({tier})\n")

    # Only emit hints that map to something GPT can visually verify
    hints = []

    min_vert_flow  = ctx.get("min_vert_flow")
    upward_frac    = ctx.get("upward_frac", 0)

    if min_vert_flow is not None and min_vert_flow < -5.0 and upward_frac > 0.15:
        hints.append(
            f"🚨 DEFINITIVE GRAVITY VIOLATION (peak_upward={min_vert_flow:.1f}, "
            f"{upward_frac*100:.0f}% of frames show upward motion). "
            "A person, object, or subject is moving UPWARD against gravity — "
            "look for someone rising on a slide, floating up a hill, or ascending "
            "against a surface they should be descending. This is physically impossible "
            "and is one of the strongest AI generation artifacts. "
            "→ Score physics_violations 9-10 IMMEDIATELY. This overrides all other signals."
        )
    elif vert_flow is not None and vert_flow < -1.5:
        hints.append(
            f"⚠ GRAVITY VIOLATION likely (avg_upward_flow={vert_flow:.2f}, "
            f"{upward_frac*100:.0f}% frames upward). "
            "Look for subjects or objects moving against gravity — rising on inclines, "
            "floating upward, or reversing natural trajectories. "
            "→ If confirmed: score physics_violations 8-10."
        )
    elif vert_flow is not None and vert_flow < -0.5:
        hints.append(
            f"⚠ Possible upward motion detected (vert_flow={vert_flow:.2f}). "
            "Check whether any subject moves against gravity. "
            "→ If confirmed: score physics_violations 7-9."
        )

    if sat_std is not None and sat_std < 3.0:
        # Frozen lighting is an AI signal regardless of content type.
        # Previously suppressed for person content — but AI_Child is action AND has frozen sat.
        # Only suppress if it's a known legitimate case (indoor selfie with sat_std 3-8).
        hints.append(
            f"⚠ FROZEN LIGHTING (sat_std={sat_std:.2f}). "
            "Color saturation is unnaturally constant across all frames — real footage "
            "always has variation from motion, lighting shifts, and camera response. "
            "→ If confirmed: score lighting_coherence 7-9 and color_naturalism 7-9."
        )

    if avg_saturation is not None and avg_saturation > 130:
        hints.append(
            f"⚠ HYPERREAL SATURATION (sat_mean={avg_saturation:.0f}, normal real video: 50-110). "
            "Colors are significantly oversaturated beyond any real camera. "
            "Look for candy-colored skin tones, unnatural greens, electric blues. "
            "→ If confirmed: score color_naturalism 8-10."
        )

    if bg_drift is not None and bg_drift < 3.0:
        hints.append(
            f"⚠ FROZEN BACKGROUND (bg_drift={bg_drift:.2f}). "
            "Background corners are nearly static — look for painted/rendered background. "
            "→ If confirmed: score background_realism 7-9."
        )

    if flicker_std is not None and flicker_std > 4.0:
        n_cuts_val = ctx.get("n_scene_cuts", 0)
        if n_cuts_val >= 3 or flicker_std > 8.0:
            # High flicker + multiple cuts = likely stitched AI compilation
            hints.append(
                f"⚠ TEMPORAL INSTABILITY (flicker_std={flicker_std:.2f}, cuts={n_cuts_val}). "
                "This video may be multiple AI-generated clips stitched together. "
                "Look for: subject/background inconsistencies between segments, "
                "subtle lighting color shifts at cuts, animal/person rendered "
                "slightly differently in each clip, environment resets between segments. "
                "Also check within each segment for AI texture shimmer/flicker. "
                "→ Score temporal_stability 7-9 if confirmed."
            )
        else:
            hints.append(
                f"⚠ FRAME FLICKER (flicker_std={flicker_std:.2f}). "
                "Compare frames for subtle texture or geometry changes between frames. "
                "→ If confirmed: score temporal_stability 7-9."
            )

    if quad_cov is not None and quad_cov < 0.40:
        hints.append(
            f"⚠ UNIFORM RENDER FOCUS (quad_cov={quad_cov:.3f}). "
            "All frame quadrants are identically sharp — real cameras have DOF variation. "
            "→ If confirmed: score background_realism 7-9."
        )

    if motion_sync is not None and motion_sync < 0.09 and not is_person:
        hints.append(
            f"⚠ LOCKSTEP MOTION (sync={motion_sync:.3f}). "
            "Left/right halves move identically — AI crowd signature. "
            "→ If confirmed: score crowd_behavior 7-9."
        )

    if flow_entropy is not None and flow_entropy < 1.5 and is_action:
        hints.append(
            f"⚠ UNIFORM CROWD MOTION (entropy={flow_entropy:.3f}). "
            "All motion vectors same direction — real action is chaotic. "
            "→ If confirmed: score crowd_behavior 7-9."
        )

    skin_ratio = ctx.get("skin_ratio", 0)
    if is_person or (skin_ratio is not None and skin_ratio > 0.10):
        hints.append(
            "✓ PERSON VISIBLE: Inspect skin_texture, hair_detail, eye_quality carefully. "
            "Real skin has pores, redness variation, fine lines. "
            "AI skin is porcelain-smooth and uniformly colored. "
            "Score 0-3 if genuinely real; score 7-9 if unnaturally perfect."
        )

    # ── AI Compilation hint ──────────────────────────────────
    n_cuts = ctx.get("n_scene_cuts", 0)
    is_comp = ctx.get("is_compilation", False)
    if is_comp and n_cuts >= 5:
        avg_clip_dur = ctx.get("avg_saturation", 0)  # use video_duration / n_cuts
        hints.append(
            (
                f"COMPILATION DETECTED ({n_cuts} scene cuts). "
                "This video is multiple short AI-generated clips stitched together. "
                "Look for: inconsistent lighting between cuts, subject appears "
                "slightly different in each clip (re-rendered), background changes "
                "subtly at cuts despite same apparent location, behavior resets "
                "unnaturally between clips. "
                "Score temporal_stability 8-10. Evaluate background_realism and "
                "physics_violations across clips, not just within one."
            )
        )

    # ── Animal / Wildlife / Pet content hint ────────────────
    # AI-generated animal videos are extremely common on TikTok/social media.
    # Generators like Sora, Kling, and Hailuo produce convincing but detectable
    # animal renders. Key tells are different from human deepfakes.
    avg_noise = ctx.get("avg_noise", 999)
    is_low_noise = avg_noise is not None and avg_noise < 300
    if content_type in ("cinematic", "action") and not is_person and skin_ratio < 0.15:
        hints.append(
            "🐾 ANIMAL / WILDLIFE CONTENT DETECTED: Apply these AI-specific checks:\n"
            "  FUR TEXTURE: Real animal fur has individual strand variation, matting, "
            "and directional flow that shifts with movement. AI fur is uniformly rendered — "
            "every strand perfectly rendered, no clumping or texture irregularity.\n"
            "  MOVEMENT PHYSICS: Real animals have weight — inertia, momentum, joint "
            "constraints. AI animals often move with uncanny smoothness or unnatural "
            "flexibility. Check paw/hoof placement, body sway, and reaction to terrain.\n"
            "  ENVIRONMENT INTERACTION: Real animals disturb their environment — "
            "grass bends underfoot, water splashes realistically, dust kicks up. "
            "AI renders often have the animal 'floating' above the environment.\n"
            "  BACKGROUND CONSISTENCY: AI animal videos frequently have background "
            "that looks rendered/painted — check for unnaturally smooth bokeh, "
            "repetitive vegetation patterns, or too-perfect depth separation.\n"
            "  EYE QUALITY: Real animal eyes have catchlights, moisture, iris detail. "
            "AI eyes are often glassy, symmetrically perfect, or anatomically off.\n"
            "→ Score skin_texture, background_realism, physics_violations carefully."
        )

    # ── High-skin-ratio creature / AI animal render hint ───
    # When the signal detector sees very high skin-range color ratio (>0.50) combined
    # with a strong motion period, this is likely an AI-generated animal/creature video.
    # Bunny, dog, cat, fox fur all trigger HSV skin detection at 0.50-0.70.
    # These are among the most common AI video types on TikTok (Kling, Hailuo, Sora).
    period_val = ctx.get("motion_period", ctx.get("period", 0))
    if skin_ratio is not None and skin_ratio > 0.50 and period_val > 0.70:
        hints.append(
            "🚨 HIGH PROBABILITY AI CREATURE/ANIMAL VIDEO "
            "(warm-color-ratio={:.2f}, motion_loop={:.3f}):\n"
            "The signal detector found very high warm-tone color coverage typical of animal fur "
            "(bunny, dog, cat, fox) combined with a strong animation loop signature. "
            "This combination is a strong indicator of AI-generated animal content "
            "from Kling, Hailuo, or Sora.\n"
            "  CRITICAL CHECKS:\n"
            "  FUR QUALITY: Zoom into the fur texture. AI fur looks perfectly rendered — "
            "every strand identical, no matting, no directional clumping from motion. "
            "Real fur has irregular texture, compression artifacts, and shifts direction "
            "naturally with movement.\n"
            "  LOOPING MOTION: AI animal videos often have subtle looping animation — "
            "the creature repeats the same motion cycle. Watch for: identical head tilt "
            "sequence, same ear flick repeating, body sway that loops. "
            "Real animals have natural variation.\n"
            "  EYE REALISM: AI animal eyes are often glassy, too symmetrical, or have "
            "unnaturally perfect catchlights. Real animal eyes have moisture, iris "
            "variation, and asymmetry.\n"
            "  BACKGROUND RENDER: AI backgrounds behind animals are often too smooth — "
            "painted-looking bokeh, repetitive foliage patterns, or unnaturally uniform "
            "depth separation.\n"
            "  ENVIRONMENT CONTACT: Does the animal touch/disturb the environment "
            "naturally? AI animals often float slightly above surfaces or leave no "
            "physical trace.\n"
            "  BODY PROPORTIONS: AI generators often slightly distort limb ratios — "
            "legs too long, ears too large, or facial features that shift between frames.\n"
            "→ Score fur as skin_texture, motion loop as temporal_stability, "
            "background as background_realism. Be AGGRESSIVE — these videos are "
            "almost always AI.\n"
            "→ If you see ANY of the above: score ai_probability 75-95."
            .format(skin_ratio, period_val)
        )

    # ── Sports / outdoor real-camera hint ──────────────────
    # Real sports/action phone videos have characteristics that look like AI signals
    # but are actually camera physics: telephoto blur (ball vs crowd), limited color
    # palette (grass + dirt + sky), and motion blur artifacts.
    # Help GPT understand when these are real physics, not AI renders.
    avg_noise_val = ctx.get("avg_noise", 0)
    if content_type in ("action", "cinematic") and avg_noise_val > 400:
        hints.append(
            "📷 REAL CAMERA INDICATORS DETECTED (noise={:.0f}): "
            "High sensor noise confirms real camera capture. "
            "In sports/outdoor video this means:\n"
            "  • Telephoto blur (ball/subject vs blurred crowd) is natural DOF, not AI render\n"
            "  • Limited color palette (grass/dirt/sky) is the real scene, not AI curation\n"
            "  • Motion blur on fast objects (balls, birds) is real shutter physics\n"
            "  • Compression artifacts from social media re-encoding are normal\n"
            "→ Only flag background_realism or color_naturalism as AI if you see "
            "RENDERED/PAINTED qualities, not just blur or palette limitation.".format(avg_noise_val)
        )

    # ── Audio mismatch hint ─────────────────────────────────
    audio_has_signal = ctx.get("audio_has_signal", False)
    audio_dur_mismatch = ctx.get("audio_dur_mismatch", 0)
    audio_stereo_corr = ctx.get("audio_stereo_corr", 0)
    audio_reason = ctx.get("audio_reason", "")

    if audio_has_signal:
        hint_parts = []
        if audio_dur_mismatch > 0.07:
            hint_parts.append(
                f"audio track is {audio_dur_mismatch:.3f}s longer than video "
                f"(typical of stock audio added after generation)"
            )
        if audio_stereo_corr > 0.87:
            hint_parts.append(
                f"stereo L/R channels are nearly identical (corr={audio_stereo_corr:.3f}) "
                f"— stock music panned to stereo, not real ambient sound"
            )
        if hint_parts:
            hints.append(
                "🎵 AI AUDIO SIGNATURES DETECTED: " + "; ".join(hint_parts) + ".\n"
                "  This strongly suggests the audio was added after video generation, "
                "not recorded with the scene.\n"
                "  → Check: does the audio match the environment? "
                "Is there natural ambient sound (wind, crowd noise, room tone)? "
                "Does the audio feel 'placed on top' rather than captured with the video?\n"
                "  → If audio feels artificial or mismatched: "
                "score temporal_stability 7-9 and color_naturalism 7-8."
            )

    if hints:
        lines.append("VISUAL INSPECTION PRIORITIES:")
        for h in hints:
            lines.append(f"  {h}")

    lines.append("\n═══════════════════════════════════════")
    lines.append("Now score all 12 dimensions based on what you observe in the frames.")
    lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
#  Public interface
# ─────────────────────────────────────────────────────────────
def gpt_vision_score(video_path: str) -> dict:
    """
    Main entry point — no signal context.
    Extract frames and run GPT-4o rubric analysis.
    Returns dict: ai_probability, scores, reasoning, flags, generator_guess.
    """
    if not OPENAI_API_KEY:
        log.warning("gpt_vision: OPENAI_API_KEY not configured")
        return _unavailable_result("GPT vision not configured")

    frames = extract_key_frames(video_path)
    if not frames:
        return _unavailable_result("Could not extract frames")

    result = analyze_frames_with_gpt(frames)
    result["available"] = True
    return result


def gpt_vision_score_with_context(frames_b64: list, physics_context: dict) -> dict:
    """
    Primary entry point — with signal detector context.
    physics_context dict contains measured signals from detector.py.
    Returns dict: ai_probability, scores, reasoning, flags, generator_guess.
    """
    if not OPENAI_API_KEY:
        return _unavailable_result("GPT vision not configured")

    if not frames_b64:
        return _unavailable_result("Could not extract frames")

    content_type    = physics_context.get("content_type", "cinematic")
    physics_summary = _build_physics_summary(physics_context)
    result          = analyze_frames_with_gpt(frames_b64, physics_summary, content_type)
    result["available"] = True
    return result


