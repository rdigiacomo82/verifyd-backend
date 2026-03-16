# ============================================================
#  VeriFYD — c2pa_checker.py
#
#  C2PA (Coalition for Content Provenance and Authenticity)
#  provenance verification engine.
#
#  C2PA manifests are cryptographically signed metadata blocks
#  embedded in media files by cameras, AI tools, and editors.
#  They record WHO created the content and WHAT tool was used.
#
#  As of 2025/2026 these tools embed C2PA:
#    REAL camera signals:
#      - Leica M11-P, Sony Alpha 9 III / Alpha 1 II
#      - Samsung Galaxy S25 (AI-edited photos)
#      - Google Pixel 10 (all photos by default)
#      - Nikon Z6 III (via firmware)
#
#    AI generation signals:
#      - OpenAI / DALL-E / Sora
#      - Adobe Firefly / Photoshop
#      - Microsoft Azure OpenAI
#      - Google DeepMind / Imagen
#
#  SCORING LOGIC:
#    - Valid C2PA manifest from known real camera → strong REAL (-20 to -30)
#    - Valid C2PA manifest from known AI tool    → strong AI (+40 to +60)
#    - C2PA manifest present but signature INVALID → mild AI (+8)
#      (stripped/tampered manifest is itself suspicious)
#    - No C2PA manifest → no signal (most videos have none yet)
#
#  WHY THIS IS SAFE TO ADD:
#    - Pure read operation, no model loading
#    - Falls back gracefully if c2pa-python not installed
#    - Uses subprocess fallback via c2patool if available
#    - Never blocks detection if it fails
#    - Zero memory overhead beyond the library itself (~5MB)
#
#  Returns:
#    c2pa_score   : int, AI contribution (0=neutral, >0=AI, <0=real)
#    c2pa_label   : str, human-readable finding
#    c2pa_detail  : dict, raw manifest data for logging/reasoning
# ============================================================

import os
import json
import logging
import subprocess
from typing import Tuple, Dict, Any

log = logging.getLogger("verifyd.c2pa")

# ─────────────────────────────────────────────────────────────
#  Known AI generator identifiers in C2PA manifests
#  Source: C2PA spec claim_generator_info.name field
# ─────────────────────────────────────────────────────────────
_AI_GENERATORS = {
    # OpenAI
    "chatgpt":          ("OpenAI ChatGPT",    55),
    "dall-e":           ("OpenAI DALL-E",     55),
    "dall·e":           ("OpenAI DALL-E",     55),
    "openai":           ("OpenAI",            50),
    "gpt-4o":           ("OpenAI GPT-4o",     55),
    # Adobe
    "adobe firefly":    ("Adobe Firefly",     55),
    "firefly":          ("Adobe Firefly",     50),
    # Google
    "imagen":           ("Google Imagen",     55),
    "gemini":           ("Google Gemini",     50),
    # Stability AI
    "stable diffusion": ("Stability AI",      50),
    "stability":        ("Stability AI",      45),
    # Runway
    "runway":           ("Runway",            55),
    "runwayml":         ("Runway",            55),
    # Sora / video models
    "sora":             ("OpenAI Sora",       60),
    "pika":             ("Pika Labs",         55),
    "kling":            ("Kuaishou Kling",    55),
    "hailuo":           ("MiniMax Hailuo",    55),
    "vidu":             ("Vidu",              55),
    # Microsoft
    "microsoft":        ("Microsoft AI",      45),
    "azure openai":     ("Azure OpenAI",      50),
    # Generic AI markers
    "ai_generated":     ("AI Generated",      55),
    "trainedAlgorithmicMedia": ("AI Generated", 55),
}

# ─────────────────────────────────────────────────────────────
#  Known real camera / authentic source identifiers
# ─────────────────────────────────────────────────────────────
_REAL_SOURCES = {
    "leica":            ("Leica Camera",     -28),
    "sony":             ("Sony Camera",      -25),
    "nikon":            ("Nikon Camera",     -25),
    "canon":            ("Canon Camera",     -25),
    "samsung galaxy":   ("Samsung Galaxy",   -20),
    "google pixel":     ("Google Pixel",     -20),
    "apple":            ("Apple iPhone",     -18),
    "iphone":           ("Apple iPhone",     -18),
    "truepic":          ("Truepic",          -22),  # photojournalism C2PA
    "witness":          ("Witness Camera",   -25),
    # News / media orgs
    "reuters":          ("Reuters",          -20),
    "ap ":              ("Associated Press", -20),
    "bbc":              ("BBC",              -18),
}

# digitalSourceType URIs that mean AI-generated
_AI_SOURCE_TYPES = {
    "trainedAlgorithmicMedia",
    "compositeSynthetic",
    "algorithmicMedia",
}

# digitalSourceType URIs that mean real camera
_REAL_SOURCE_TYPES = {
    "digitalCapture",
    "negativeFilm",
    "positiveFilm",
    "print",
    "screenCapture",
}


def check_c2pa(video_path: str) -> Tuple[int, str, Dict[str, Any]]:
    """
    Check video file for C2PA Content Credentials.

    Returns:
        (score_delta, label, detail_dict)
        score_delta: added to AI score (+= more AI, -= more real)
        label: human-readable finding for logging
        detail_dict: manifest data for reasoning builder
    """
    detail: Dict[str, Any] = {
        "has_manifest":    False,
        "manifest_valid":  False,
        "generator":       None,
        "source_type":     None,
        "issuer":          None,
        "actions":         [],
        "score_delta":     0,
        "label":           "no_c2pa",
    }

    # Try c2pa-python library first (preferred)
    try:
        import c2pa
        result = _check_via_library(video_path, c2pa)
        if result is not None:
            return result
    except ImportError:
        log.debug("c2pa-python not installed — trying c2patool fallback")
    except Exception as e:
        log.warning("c2pa library error: %s", e)

    # Fallback: c2patool CLI (if installed on system)
    try:
        result = _check_via_cli(video_path)
        if result is not None:
            return result
    except Exception as e:
        log.debug("c2patool fallback failed: %s", e)

    # No C2PA data found or tools unavailable
    log.debug("C2PA: no manifest found in %s", os.path.basename(video_path))
    return 0, "no_c2pa", detail


def _check_via_library(video_path: str, c2pa_module) -> Tuple[int, str, Dict] | None:
    """Use c2pa-python library to read manifest."""
    try:
        # c2pa-python 0.28+ API
        reader = c2pa_module.Reader(video_path)
        manifest_store_json = reader.json()
        if not manifest_store_json:
            return None

        store = json.loads(manifest_store_json)
        return _parse_manifest_store(store)

    except Exception as e:
        log.debug("c2pa library read failed: %s", e)
        return None


def _check_via_cli(video_path: str) -> Tuple[int, str, Dict] | None:
    """Fallback: use c2patool CLI if installed."""
    try:
        result = subprocess.run(
            ["c2patool", video_path, "--output", "/dev/stdout"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None

        store = json.loads(result.stdout)
        return _parse_manifest_store(store)

    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    except json.JSONDecodeError:
        return None


def _parse_manifest_store(store: dict) -> Tuple[int, str, Dict]:
    """
    Parse C2PA manifest store JSON and return score delta + detail.
    """
    detail: Dict[str, Any] = {
        "has_manifest":   True,
        "manifest_valid": False,
        "generator":      None,
        "source_type":    None,
        "issuer":         None,
        "actions":        [],
        "score_delta":    0,
        "label":          "manifest_found",
    }

    # Check validation status
    validation_status = store.get("validation_status", [])
    has_errors = any(
        s.get("code", "").startswith("claimSignature") or
        s.get("code", "").startswith("assertion.dataHash")
        for s in (validation_status if isinstance(validation_status, list) else [])
    )

    if has_errors:
        # Manifest present but signature broken — suspicious
        log.info("C2PA: invalid/tampered manifest → mild AI signal +8")
        detail["manifest_valid"] = False
        detail["label"] = "manifest_invalid"
        detail["score_delta"] = 8
        return 8, "manifest_invalid_tampered", detail

    detail["manifest_valid"] = True

    # Get active manifest
    active_key = store.get("active_manifest")
    manifests = store.get("manifests", {})
    manifest = manifests.get(active_key, {}) if active_key else {}
    if not manifest and manifests:
        manifest = list(manifests.values())[-1]

    if not manifest:
        return 0, "manifest_empty", detail

    # Extract claim generator info
    generators = manifest.get("claim_generator_info", [])
    for gen in generators if isinstance(generators, list) else []:
        name = str(gen.get("name", "")).lower()
        detail["generator"] = gen.get("name")

        # Check AI generators
        for key, (label, score) in _AI_GENERATORS.items():
            if key in name:
                log.info("C2PA: AI generator detected → %s → +%d", label, score)
                detail["label"] = f"ai_generator_{label.replace(' ', '_')}"
                detail["score_delta"] = score
                return score, f"c2pa_ai_{label}", detail

        # Check real sources
        for key, (label, score) in _REAL_SOURCES.items():
            if key in name:
                log.info("C2PA: real source detected → %s → %d", label, score)
                detail["label"] = f"real_source_{label.replace(' ', '_')}"
                detail["score_delta"] = score
                return score, f"c2pa_real_{label}", detail

    # Check assertions for digitalSourceType
    assertions = manifest.get("assertions", [])
    for assertion in assertions if isinstance(assertions, list) else []:
        data = assertion.get("data", {})

        # c2pa.actions assertion
        if assertion.get("label", "").startswith("c2pa.actions"):
            actions = data.get("actions", [])
            for action in actions if isinstance(actions, list) else []:
                detail["actions"].append(action.get("action", ""))

                # Check softwareAgent
                agent = action.get("softwareAgent", {})
                if isinstance(agent, dict):
                    agent_name = str(agent.get("name", "")).lower()
                    for key, (label, score) in _AI_GENERATORS.items():
                        if key in agent_name:
                            log.info("C2PA: AI action agent → %s → +%d", label, score)
                            detail["generator"] = agent.get("name")
                            detail["score_delta"] = score
                            detail["label"] = f"ai_action_{label}"
                            return score, f"c2pa_ai_action_{label}", detail

                # Check digitalSourceType
                dst = action.get("digitalSourceType", "")
                if isinstance(dst, str):
                    dst_short = dst.split("/")[-1]
                    detail["source_type"] = dst_short
                    if dst_short in _AI_SOURCE_TYPES:
                        log.info("C2PA: AI digitalSourceType → %s → +50", dst_short)
                        detail["score_delta"] = 50
                        detail["label"] = f"ai_source_type_{dst_short}"
                        return 50, f"c2pa_ai_source_{dst_short}", detail
                    if dst_short in _REAL_SOURCE_TYPES:
                        log.info("C2PA: real digitalSourceType → %s → -20", dst_short)
                        detail["score_delta"] = -20
                        detail["label"] = f"real_source_type_{dst_short}"
                        return -20, f"c2pa_real_capture", detail

    # Signature issuer
    sig_info = manifest.get("signature_info", {})
    issuer = sig_info.get("issuer", "")
    detail["issuer"] = issuer
    if issuer:
        issuer_lower = issuer.lower()
        for key, (label, score) in _AI_GENERATORS.items():
            if key in issuer_lower:
                log.info("C2PA: AI issuer → %s → +%d", label, score)
                detail["score_delta"] = score
                detail["label"] = f"ai_issuer_{label}"
                return score, f"c2pa_ai_issuer_{label}", detail
        for key, (label, score) in _REAL_SOURCES.items():
            if key in issuer_lower:
                log.info("C2PA: real issuer → %s → %d", label, score)
                detail["score_delta"] = score
                detail["label"] = f"real_issuer_{label}"
                return score, f"c2pa_real_issuer_{label}", detail

    # Valid manifest but unknown source — mild real signal
    # (someone bothered to sign it, likely a real tool)
    log.info("C2PA: valid manifest, unknown source → mild real signal -5")
    detail["score_delta"] = -5
    detail["label"] = "manifest_valid_unknown"
    return -5, "c2pa_valid_unknown_source", detail
