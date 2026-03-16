# ============================================================
#  VeriFYD — dinov2_detector.py
#
#  DINOv2 ViT-Small Feature-Based AI Video Detection
#
#  WHY DINOV2 WORKS FOR AI DETECTION:
#  DINOv2 was trained on 142M real-world images using self-
#  supervised learning. Its patch-level features capture rich
#  texture and structure information. AI-generated frames have
#  fundamentally different feature distributions:
#
#  1. PATCH FEATURE UNIFORMITY
#     Real camera frames: high variance across patches (random
#     sensor noise, natural texture variation, lens aberration)
#     AI frames: lower variance (upsampling creates smooth,
#     regular patch patterns — the VAE decoder is too uniform)
#
#  2. ATTENTION MAP ENTROPY
#     DINOv2 attention heads focus on semantically meaningful
#     regions in real images (edges, objects, texture boundaries)
#     In AI frames, attention is more diffuse/uniform because
#     AI generators produce globally coherent but locally
#     smooth regions
#
#  3. CLS TOKEN SIMILARITY TO PATCH TOKENS
#     In real images, the CLS token (global representation)
#     differs significantly from individual patch tokens
#     (local representations). In AI images, the CLS token
#     is more similar to patches because AI generators lack
#     the camera sensor's random local variation.
#
#  4. INTER-FRAME FEATURE CONSISTENCY (VIDEO SPECIFIC)
#     Real video: patch features vary randomly frame-to-frame
#     due to sensor noise. AI video: patch features are highly
#     consistent across frames (same generator artifacts repeat)
#
#  MEMORY BUDGET:
#    DINOv2 ViT-S/14: 21.7M params = ~83MB weights (FP32)
#    With activations at inference: ~350MB total
#    Safe on 4GB Pro tier with 2.5GB remaining for other engines
#
#  LOADING STRATEGY:
#    Model loaded ONCE at worker startup and kept in memory.
#    Uses torch.no_grad() + CPU inference (no GPU needed).
#    Lazy import — if torch not available, returns 0 gracefully.
#
#  CONTRIBUTION TO FINAL SCORE:
#    Max ±15 points. Conservative weight since this is a
#    supplementary engine, not the primary detector.
#    Only contributes when signal detector score is ambiguous
#    (40-70 range) to break ties, not to override strong signals.
#
#  Returns:
#    dino_score   : int 0-100, AI probability
#    dino_signals : dict, individual metrics for logging
# ============================================================

import os
import logging
import numpy as np
from typing import Tuple, Dict, List, Optional

log = logging.getLogger("verifyd.dinov2")

# Global model cache — loaded once at first use
_dino_model = None
_dino_loaded = False
_dino_available = None  # None=unknown, True=available, False=unavailable

_MAX_FRAMES   = 8     # frames per clip for DINOv2 analysis
_PATCH_SIZE   = 14    # DINOv2 ViT-S/14 patch size
_IMG_SIZE     = 224   # input resolution (14*16=224)


def _load_model():
    """Load DINOv2 ViT-Small model — called once at first use."""
    global _dino_model, _dino_loaded, _dino_available

    if _dino_loaded:
        return _dino_model is not None

    _dino_loaded = True

    try:
        import torch
        from transformers import AutoModel, AutoImageProcessor

        log.info("DINOv2: loading ViT-Small model...")

        # Use HuggingFace transformers for reliable loading
        model_name = "facebook/dinov2-small"
        processor = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        processor.eval()

        # Store as tuple (model, None) — processor built manually
        _dino_model = processor
        _dino_available = True
        log.info("DINOv2: ViT-Small loaded successfully (~350MB)")
        return True

    except ImportError:
        log.warning("DINOv2: transformers/torch not installed — skipping")
        _dino_available = False
        return False
    except Exception as e:
        log.warning("DINOv2: model load failed (%s) — skipping", e)
        _dino_available = False
        return False


def _preprocess_frame(frame_bgr: np.ndarray) -> Optional["torch.Tensor"]:
    """Convert BGR frame to DINOv2 input tensor."""
    try:
        import torch
        import cv2

        # Resize to 224x224
        frame = cv2.resize(frame_bgr, (_IMG_SIZE, _IMG_SIZE))
        # BGR → RGB, normalize to [0,1]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # ImageNet normalization (DINOv2 standard)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        frame_norm = (frame_rgb - mean) / std
        # HWC → CHW, add batch dim
        tensor = torch.from_numpy(frame_norm.transpose(2, 0, 1)).unsqueeze(0)
        return tensor
    except Exception:
        return None


def _extract_features(frames_bgr: List[np.ndarray]) -> Optional[Dict]:
    """
    Run DINOv2 on frames and extract feature statistics.
    Returns dict with patch features, CLS token, attention stats.
    """
    try:
        import torch

        model = _dino_model
        all_cls      = []
        all_patch_vars  = []
        all_cls_patch_sims = []

        with torch.no_grad():
            for frame in frames_bgr:
                tensor = _preprocess_frame(frame)
                if tensor is None:
                    continue

                # Forward pass — get hidden states
                outputs = model(
                    pixel_values=tensor,
                    output_hidden_states=True,
                    return_dict=True,
                )

                # Last hidden state: [1, num_patches+1, embed_dim]
                # Patch tokens: [1, num_patches, embed_dim] (skip CLS at index 0)
                last_hidden = outputs.last_hidden_state  # [1, 257, 384] for ViT-S
                cls_token   = last_hidden[:, 0, :]       # [1, 384]
                patch_tokens = last_hidden[:, 1:, :]     # [1, 256, 384]

                # 1. Patch feature variance
                # Real: high variance (diverse textures)
                # AI: lower variance (smooth upsampled regions)
                patch_var = float(patch_tokens.var(dim=1).mean().item())
                all_patch_vars.append(patch_var)

                # 2. CLS-patch similarity
                # Normalize for cosine similarity
                cls_norm   = cls_token / (cls_token.norm(dim=-1, keepdim=True) + 1e-8)
                patch_norm = patch_tokens / (patch_tokens.norm(dim=-1, keepdim=True) + 1e-8)
                # Mean cosine similarity between CLS and each patch
                cos_sims = (patch_norm * cls_norm.unsqueeze(1)).sum(dim=-1)  # [1, 256]
                cls_patch_sim = float(cos_sims.mean().item())
                all_cls_patch_sims.append(cls_patch_sim)

                all_cls.append(cls_token.squeeze(0).numpy())

        if not all_patch_vars:
            return None

        # Inter-frame CLS consistency (for video)
        cls_consistency = 0.0
        if len(all_cls) >= 2:
            cls_array = np.stack(all_cls)  # [n_frames, 384]
            # Pairwise cosine similarities between consecutive frames
            sims = []
            for i in range(len(cls_array) - 1):
                a = cls_array[i] / (np.linalg.norm(cls_array[i]) + 1e-8)
                b = cls_array[i+1] / (np.linalg.norm(cls_array[i+1]) + 1e-8)
                sims.append(float(np.dot(a, b)))
            cls_consistency = float(np.mean(sims))

        return {
            "patch_var":       float(np.mean(all_patch_vars)),
            "cls_patch_sim":   float(np.mean(all_cls_patch_sims)),
            "cls_consistency": cls_consistency,
            "n_frames":        len(all_patch_vars),
        }

    except Exception as e:
        log.debug("DINOv2 feature extraction error: %s", e)
        return None


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
    except Exception:
        return []


def analyze_dinov2(video_path: str) -> Tuple[int, Dict]:
    """
    Run DINOv2 feature analysis on a video clip.

    Returns:
        (dino_ai_score, signals_dict)
        dino_ai_score: 0-100, higher = more likely AI
    """
    signals: Dict = {
        "patch_var":       None,
        "cls_patch_sim":   None,
        "cls_consistency": None,
        "dino_score":      0,
        "available":       False,
    }

    # Load model on first call
    if not _load_model():
        return 0, signals

    signals["available"] = True

    try:
        frames = _extract_frames(video_path, _MAX_FRAMES)
        if len(frames) < 2:
            log.debug("DINOv2: insufficient frames (%d)", len(frames))
            return 0, signals

        feats = _extract_features(frames)
        if feats is None:
            return 0, signals

        patch_var      = feats["patch_var"]
        cls_patch_sim  = feats["cls_patch_sim"]
        cls_consistency = feats["cls_consistency"]

        signals["patch_var"]       = round(patch_var, 4)
        signals["cls_patch_sim"]   = round(cls_patch_sim, 4)
        signals["cls_consistency"] = round(cls_consistency, 4)

        score = 0
        components = []

        # ── Signal 1: Patch feature variance ─────────────────
        # Real camera: high variance due to sensor noise + texture
        # AI generated: lower variance (smooth upsampled regions)
        # Calibrated from test videos:
        #   Real (cow, baseball, golf): patch_var ~0.15-0.35
        #   AI (TikTok AI videos):      patch_var ~0.05-0.12
        if patch_var < 0.06:
            pv_score = 20
        elif patch_var < 0.09:
            pv_score = 14
        elif patch_var < 0.12:
            pv_score = 8
        elif patch_var < 0.15:
            pv_score = 3
        elif patch_var > 0.30:
            pv_score = -6   # Strong real signal
        elif patch_var > 0.22:
            pv_score = -3
        else:
            pv_score = 0
        score += pv_score
        components.append(f"patch_var={patch_var:.3f}→{pv_score:+d}")

        # ── Signal 2: CLS-patch similarity ───────────────────
        # Real: CLS diverges from patches (global ≠ local)
        # AI: CLS more similar to patches (global ≈ local, less diversity)
        # Higher similarity → more AI
        if cls_patch_sim > 0.45:
            cp_score = 15
        elif cls_patch_sim > 0.38:
            cp_score = 10
        elif cls_patch_sim > 0.30:
            cp_score = 5
        elif cls_patch_sim < 0.15:
            cp_score = -5  # Real signal
        elif cls_patch_sim < 0.20:
            cp_score = -2
        else:
            cp_score = 0
        score += cp_score
        components.append(f"cls_sim={cls_patch_sim:.3f}→{cp_score:+d}")

        # ── Signal 3: Inter-frame CLS consistency ────────────
        # Real video: scene changes + noise → lower CLS consistency
        # AI video: generator produces consistent global features
        # across frames (same style, same rendering artifacts)
        # CALIBRATION NOTE: Real static-subject videos (talking heads,
        # single subjects) legitimately score 0.90-0.96 because the
        # subject doesn't change. Only flag at very high thresholds.
        if cls_consistency > 0.97:
            cc_score = 15   # Near-perfect consistency → AI generator
        elif cls_consistency > 0.95:
            cc_score = 8    # Very high → suspicious but not definitive
        elif cls_consistency > 0.93:
            cc_score = 3    # Slightly elevated
        elif cls_consistency < 0.55:
            cc_score = -5   # Very low → lots of natural scene variation
        elif cls_consistency < 0.65:
            cc_score = -2
        else:
            cc_score = 0
        score += cc_score
        components.append(f"cls_cons={cls_consistency:.3f}→{cc_score:+d}")

        score = max(0, min(100, score))
        signals["dino_score"] = score

        log.info("DINOv2: score=%d [%s]", score, " ".join(components))
        return score, signals

    except Exception as e:
        log.warning("DINOv2 analysis failed: %s", e)
        return 0, signals


def get_dino_contribution(dino_score: int, signal_score: int) -> int:
    """
    Convert DINOv2 score to contribution for main AI score.

    DINOv2 is a TIE-BREAKER engine — it only contributes meaningfully
    when the signal detector is ambiguous (40-70 range).
    When signal is already confident, DINOv2 adds a smaller boost.

    Max contribution: +12 (AI) or -8 (Real)
    """
    # Only strong DINOv2 signals contribute when signal is ambiguous
    signal_ambiguous = 40 <= signal_score <= 70

    if dino_score >= 70:
        return 12 if signal_ambiguous else 6
    elif dino_score >= 55:
        return 8 if signal_ambiguous else 4
    elif dino_score >= 40:
        return 4 if signal_ambiguous else 2
    elif dino_score <= 15:
        return -8 if signal_ambiguous else -4  # Real signal
    elif dino_score <= 25:
        return -4 if signal_ambiguous else -2
    return 0
