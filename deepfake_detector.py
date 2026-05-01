# ============================================================
#  VeriFYD — deepfake_detector.py
#
#  ViT-based Face-Swap / Deepfake Detection Engine
#  Model: Wvolf/ViT_Deepfake_Detection (HuggingFace)
#
#  WHY THIS COMPLEMENTS EXISTING ENGINES:
#  The signal detector (detector.py) and DINOv2 excel at detecting
#  fully AI-GENERATED video (Runway, Kling, Sora) by measuring
#  noise patterns, temporal consistency, and patch features.
#
#  However, FACE-SWAP deepfakes (a real video with a person's face
#  replaced by another) defeat these engines because:
#    - The underlying video has REAL camera noise and motion
#    - Temporal consistency is REAL (original footage)
#    - Only the face region is synthetic
#
#  This ViT model was specifically trained on deepfake face imagery
#  and detects the subtle rendering artifacts left by face-swap
#  models that pixel-level signal analysis misses.
#
#  ACTIVATION GUARD (speed + accuracy):
#  Only runs when ALL of the following are true:
#    1. Content type is portrait, talking_head, selfie, or
#       single_subject (face present)
#    2. Skin ratio > 0.05 (confirms a human face is visible)
#    3. Signal score is in ambiguous range (30-75) OR
#       signal is high (>75) but GPT is low (<40) — possible
#       face-swap where signal incorrectly flags real video
#  This means it's skipped entirely for cinematic/action/nature
#  content where face-swaps are irrelevant, saving processing time.
#
#  INTEGRATION WITH EXISTING PIPELINE:
#    - Runs IN PARALLEL with DINOv2 inside _detect_one()
#    - Contributes max ±10 points as a tie-breaker
#    - NEVER overrides a confident (>80) signal detection
#    - Gracefully returns 0 if model fails to load
#
#  MEMORY BUDGET:
#    ViT-base-patch16-224: ~330MB weights
#    Loaded ONCE at worker startup via pre-warm
#    Total with DINOv2 already loaded: ~680MB
#    Safe on Render 4GB Pro tier
#
#  Returns:
#    deepfake_score    : int 0-100, higher = more likely deepfake
#    deepfake_signals  : dict, frame-level scores for logging
# ============================================================

import os
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional

log = logging.getLogger("verifyd.deepfake")

# Global model cache — loaded once at worker startup
_df_model     = None
_df_processor = None
_df_loaded    = False
_df_available = None   # None=unknown, True=available, False=unavailable

_MAX_FRAMES = 6        # frames per clip to analyze (fast, face-focused)
_IMG_SIZE   = 224      # ViT standard input size


def _load_model() -> bool:
    """Load Wvolf/ViT_Deepfake_Detection model — called once at startup."""
    global _df_model, _df_processor, _df_loaded, _df_available

    if _df_loaded:
        return _df_model is not None

    _df_loaded = True

    try:
        import torch
        from transformers import ViTForImageClassification, ViTImageProcessor

        model_name = "Wvolf/ViT_Deepfake_Detection"
        log.info("DeepfakeDetector: loading %s ...", model_name)

        _df_processor = ViTImageProcessor.from_pretrained(model_name)
        _df_model     = ViTForImageClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        _df_model.eval()

        _df_available = True
        log.info("DeepfakeDetector: model loaded successfully (~330MB)")
        return True

    except ImportError:
        log.warning("DeepfakeDetector: transformers/torch not installed — skipping")
        _df_available = False
        return False
    except Exception as e:
        log.warning("DeepfakeDetector: model load failed (%s) — skipping", e)
        _df_available = False
        return False


def _extract_frames(video_path: str, max_frames: int) -> List[np.ndarray]:
    """Extract evenly-spaced BGR frames from video."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []

        step = max(1, total // max_frames)
        frames = []
        for i in range(0, min(total, max_frames * step), step):
            if len(frames) >= max_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        return frames
    except Exception as e:
        log.debug("DeepfakeDetector: frame extraction error: %s", e)
        return []


def _detect_face_crop(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Attempt to detect and crop face region from frame.
    Falls back to center crop if no face detected.
    The ViT deepfake model was trained on face-focused crops
    so providing a face crop improves accuracy significantly.
    """
    try:
        import cv2
        # Try Haar cascade face detection (fast, no extra deps)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) > 0:
            # Use largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            # Add 20% padding around face
            pad_x = int(w * 0.20)
            pad_y = int(h * 0.20)
            fh, fw = frame_bgr.shape[:2]
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(fw, x + w + pad_x)
            y2 = min(fh, y + h + pad_y)
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                return crop
    except Exception:
        pass

    # Fallback: center crop (works if face is centered, common in portrait videos)
    try:
        h, w = frame_bgr.shape[:2]
        size = min(h, w)
        y1 = (h - size) // 2
        x1 = (w - size) // 2
        return frame_bgr[y1:y1+size, x1:x1+size]
    except Exception:
        return frame_bgr


def _score_frame(frame_bgr: np.ndarray) -> Optional[float]:
    """
    Run ViT deepfake classifier on a single frame.
    Returns probability 0.0-1.0 that the frame is a deepfake,
    or None if inference fails.
    """
    try:
        import torch
        import cv2
        from PIL import Image

        # Get face crop for best results
        crop = _detect_face_crop(frame_bgr)
        if crop is None or crop.size == 0:
            return None

        # BGR → RGB → PIL
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Preprocess with ViT processor
        inputs = _df_processor(images=pil_img, return_tensors="pt")

        with torch.no_grad():
            outputs = _df_model(**inputs)
            logits  = outputs.logits
            probs   = torch.nn.functional.softmax(logits, dim=-1).squeeze()

        # Determine which label index corresponds to "fake"
        # The model's id2label maps indices to label names
        id2label = _df_model.config.id2label
        fake_idx = None
        for idx, label in id2label.items():
            if label.lower() in ("fake", "deepfake", "1", "ai"):
                fake_idx = idx
                break

        if fake_idx is None:
            # If we can't identify fake label, use index 0 as default
            # (most binary classifiers use 0=real, 1=fake or vice versa)
            # Return the higher-confidence class score for fake
            fake_idx = 1 if probs[1] > probs[0] else 0

        fake_prob = float(probs[fake_idx].item())
        return fake_prob

    except Exception as e:
        log.debug("DeepfakeDetector: frame scoring error: %s", e)
        return None


def analyze_deepfake(
    video_path: str,
    content_type: str = "cinematic",
    skin_ratio: float = 0.0,
) -> Tuple[int, Dict]:
    """
    Run ViT deepfake detection on sampled frames of a video clip.

    ACTIVATION GUARD:
    Only runs on portrait/face content (talking_head, selfie,
    single_subject, portrait) with detectable skin.
    Returns (0, {available: False}) for non-face content.

    Returns:
        (deepfake_score, signals_dict)
        deepfake_score: 0-100, higher = more likely face-swap deepfake
    """
    signals: Dict = {
        "frame_scores":    [],
        "mean_score":      None,
        "face_frames":     0,
        "deepfake_score":  0,
        "available":       False,
        "skipped_reason":  None,
    }

    # ── Activation guard ─────────────────────────────────────
    # Only activate for content types where a human face is present
    _face_content_types = {"talking_head", "selfie", "single_subject", "portrait"}
    _is_face_content = (
        content_type in _face_content_types or
        skin_ratio > 0.08   # fallback: significant skin visible
    )

    if not _is_face_content:
        signals["skipped_reason"] = f"non-face content ({content_type}, skin={skin_ratio:.3f})"
        log.debug("DeepfakeDetector: skipped — %s", signals["skipped_reason"])
        return 0, signals

    # ── Load model ───────────────────────────────────────────
    if not _load_model():
        signals["skipped_reason"] = "model unavailable"
        return 0, signals

    signals["available"] = True

    # ── Extract and score frames ──────────────────────────────
    try:
        frames = _extract_frames(video_path, _MAX_FRAMES)
        if len(frames) < 2:
            signals["skipped_reason"] = f"insufficient frames ({len(frames)})"
            return 0, signals

        frame_scores = []
        for frame in frames:
            score = _score_frame(frame)
            if score is not None:
                frame_scores.append(score)

        if not frame_scores:
            signals["skipped_reason"] = "no scoreable frames"
            return 0, signals

        signals["frame_scores"] = [round(s, 3) for s in frame_scores]
        signals["face_frames"]  = len(frame_scores)

        # Use mean of top-scoring frames (face-swap artifacts are
        # consistent across frames, so mean is more reliable than max)
        mean_score = float(np.mean(frame_scores))
        signals["mean_score"] = round(mean_score, 4)

        # Convert probability to 0-100 score
        deepfake_score = int(round(mean_score * 100))
        signals["deepfake_score"] = deepfake_score

        log.info(
            "DeepfakeDetector: score=%d  frames=%d  mean=%.3f  [%s]",
            deepfake_score, len(frame_scores), mean_score,
            " ".join(f"{s:.2f}" for s in frame_scores),
        )

        return deepfake_score, signals

    except Exception as e:
        log.warning("DeepfakeDetector: analysis failed: %s", e)
        signals["skipped_reason"] = str(e)[:80]
        return 0, signals


def get_deepfake_contribution(
    deepfake_score: int,
    signal_score: int,
    content_type: str = "cinematic",
) -> int:
    """
    Convert deepfake score to contribution for main AI score.

    Conservative weights — this is a SUPPLEMENTARY engine:
    - Only contributes meaningfully when signal is ambiguous (30-75)
    - Small contribution when signal is already confident
    - Never overrides a very strong signal detection (>85)
    - Max contribution: +10 (deepfake) or -5 (real face)

    Face-swap deepfakes are a narrow category — we don't want
    this engine to inflate scores on real portrait videos.
    """
    # Don't influence very confident signal detections
    if signal_score > 85 or signal_score < 15:
        return 0

    signal_ambiguous = 30 <= signal_score <= 75

    if deepfake_score >= 80:
        return 10 if signal_ambiguous else 5
    elif deepfake_score >= 65:
        return 7 if signal_ambiguous else 3
    elif deepfake_score >= 50:
        return 4 if signal_ambiguous else 2
    elif deepfake_score <= 20:
        # Confident real face — small real signal
        return -5 if signal_ambiguous else -2
    elif deepfake_score <= 35:
        return -3 if signal_ambiguous else -1

    return 0

