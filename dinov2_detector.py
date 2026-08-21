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


# ============================================================
# VERIFYD_RESTRAV_SHADOW_V1
#
# ReStraV-inspired temporal geometry using the EXISTING DINOv2
# ViT-S/14 model already loaded by VeriFYD.
#
# Safety:
#   - shadow / observability mode only
#   - does NOT change dino_score
#   - does NOT change get_dino_contribution()
#   - does NOT change final VeriFYD classification
#   - no new model and no new dependency
#
# Official ReStraV geometry:
#   24 frames from an approximately 2-second window
#   7 early step distances
#   6 early turning angles
#   8 aggregate statistics
#   = 21 features
#
# Enable on Render with:
#   VERIFYD_RESTRAV_SHADOW_MODE=true
# ============================================================

_RESTRAV_FRAMES = 24
_RESTRAV_WINDOW_SEC = 2.0


def _restrav_shadow_enabled() -> bool:
    return os.environ.get(
        "VERIFYD_RESTRAV_SHADOW_MODE", "false"
    ).strip().lower() in ("1", "true", "yes", "on")



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




def _extract_restrav_frames(
    video_path: str,
    n_frames: int = _RESTRAV_FRAMES,
    window_sec: float = _RESTRAV_WINDOW_SEC,
) -> List[np.ndarray]:
    """
    Sample a centered temporal window for ReStraV shadow analysis.

    Mirrors the published ReStraV sampling concept:
      - approximately 2 seconds
      - 24 regularly spaced frames

    Uses OpenCV so VeriFYD does not need torchcodec or any new dependency.
    """
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        if total <= 0:
            cap.release()
            return []

        if fps <= 0.1:
            fps = 30.0

        duration = total / fps

        if duration <= 0:
            cap.release()
            return []

        effective_window = min(float(window_sec), duration)
        center_time = duration / 2.0

        start_time = max(0.0, center_time - effective_window / 2.0)
        end_time = min(duration, center_time + effective_window / 2.0)

        # Avoid selecting exactly past the final decodable frame.
        if end_time >= duration:
            end_time = max(start_time, duration - (1.0 / fps))

        times = np.linspace(
            start_time,
            end_time,
            num=max(8, int(n_frames)),
            dtype=np.float64,
        )

        frames = []

        for t in times:
            frame_index = int(round(float(t) * fps))
            frame_index = max(0, min(total - 1, frame_index))

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()

            if ret and frame is not None:
                frames.append(frame)

        cap.release()

        return frames

    except Exception as exc:
        log.debug("ReStraV shadow frame extraction failed: %s", exc)
        return []


def _extract_restrav_geometry(
    frames_bgr: List[np.ndarray],
) -> Optional[Dict]:
    """
    Compute ReStraV temporal geometry from DINOv2 token trajectories.

    The published method treats each frame's full DINOv2 token representation
    as a point in representation space.

    For consecutive frames:
      delta_t = Z[t+1] - Z[t]
      d_t     = ||delta_t||
      theta_t = angle(delta_t, delta_t+1)

    Returned official-style 21-D feature vector:
      d[0:7]                       -> 7
      theta[0:6]                   -> 6
      mean/min/max/variance(d)     -> 4
      mean/min/max/variance(theta) -> 4
                                      --
                                      21

    This function intentionally does NOT classify the features.
    """
    if len(frames_bgr) < 8:
        return None

    try:
        import torch

        model = _dino_model
        trajectory = []

        with torch.no_grad():
            for frame in frames_bgr:
                tensor = _preprocess_frame(frame)

                if tensor is None:
                    continue

                outputs = model(
                    pixel_values=tensor,
                    output_hidden_states=False,
                    return_dict=True,
                )

                # HuggingFace DINOv2:
                # [CLS token + patch tokens, embedding dimension]
                tokens = outputs.last_hidden_state.squeeze(0)

                # ReStraV treats the complete token representation as one
                # frame-level representation-space point.
                flat_tokens = (
                    tokens
                    .reshape(-1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )

                trajectory.append(flat_tokens)

        if len(trajectory) < 8:
            return None

        Z = np.stack(trajectory, axis=0).astype(np.float32, copy=False)

        delta = Z[1:] - Z[:-1]

        distances = np.linalg.norm(
            delta.astype(np.float64),
            axis=1,
        )

        if len(distances) < 7 or len(delta) < 7:
            return None

        a = delta[:-1].astype(np.float64)
        b = delta[1:].astype(np.float64)

        numerator = np.sum(a * b, axis=1)
        denominator = (
            np.linalg.norm(a, axis=1)
            * np.linalg.norm(b, axis=1)
            + 1e-12
        )

        cosines = np.clip(
            numerator / denominator,
            -1.0,
            1.0,
        )

        angles = np.degrees(np.arccos(cosines))

        if len(angles) < 6:
            return None

        stats = np.array([
            float(np.mean(distances)),
            float(np.min(distances)),
            float(np.max(distances)),
            float(np.var(distances)),
            float(np.mean(angles)),
            float(np.min(angles)),
            float(np.max(angles)),
            float(np.var(angles)),
        ], dtype=np.float64)

        feature_vector = np.concatenate([
            distances[:7],
            angles[:6],
            stats,
        ])

        if feature_vector.shape[0] != 21:
            return None

        # Additional interpretable shadow diagnostics.
        path_length = float(np.sum(distances))

        endpoint_distance = float(
            np.linalg.norm(
                (Z[-1] - Z[0]).astype(np.float64)
            )
        )

        straightness_ratio = (
            endpoint_distance / path_length
            if path_length > 1e-12
            else 0.0
        )

        step_mean = float(np.mean(distances))
        step_std = float(np.std(distances))

        step_cv = (
            step_std / step_mean
            if step_mean > 1e-12
            else 0.0
        )

        return {
            "available": True,
            "n_frames": len(trajectory),

            # Official-style 21-D temporal geometry vector.
            "feature_vector": [
                round(float(x), 6)
                for x in feature_vector.tolist()
            ],

            # Easier-to-read shadow telemetry.
            "mean_step_distance": round(float(np.mean(distances)), 6),
            "min_step_distance": round(float(np.min(distances)), 6),
            "max_step_distance": round(float(np.max(distances)), 6),
            "step_variance": round(float(np.var(distances)), 6),
            "step_cv": round(float(step_cv), 6),

            "mean_turn_angle_deg": round(float(np.mean(angles)), 6),
            "min_turn_angle_deg": round(float(np.min(angles)), 6),
            "max_turn_angle_deg": round(float(np.max(angles)), 6),
            "turn_angle_variance": round(float(np.var(angles)), 6),

            "path_length": round(path_length, 6),
            "endpoint_distance": round(endpoint_distance, 6),
            "straightness_ratio": round(float(straightness_ratio), 6),

            # Explicit rollout protection.
            "classification_enabled": False,
            "score_contribution": 0,
        }

    except Exception as exc:
        log.debug("ReStraV shadow geometry failed: %s", exc)
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

        # VERIFYD_RESTRAV_SHADOW_V1
        "restrav_shadow_enabled": _restrav_shadow_enabled(),
        "restrav_available": False,
        "restrav_n_frames": 0,
        "restrav_feature_vector": [],
        "restrav_mean_step_distance": None,
        "restrav_mean_turn_angle_deg": None,
        "restrav_straightness_ratio": None,
        "restrav_score_contribution": 0,
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

        # ====================================================
        # VERIFYD_RESTRAV_SHADOW_V1
        #
        # Collect ReStraV temporal geometry for benchmarking.
        # It MUST NOT alter score or production classification.
        # ====================================================
        if _restrav_shadow_enabled():
            try:
                restrav_frames = _extract_restrav_frames(
                    video_path,
                    n_frames=_RESTRAV_FRAMES,
                    window_sec=_RESTRAV_WINDOW_SEC,
                )

                restrav = _extract_restrav_geometry(restrav_frames)

                if restrav:
                    signals["restrav_available"] = True
                    signals["restrav_n_frames"] = restrav.get("n_frames", 0)
                    signals["restrav_feature_vector"] = restrav.get(
                        "feature_vector", []
                    )

                    signals["restrav_mean_step_distance"] = restrav.get(
                        "mean_step_distance"
                    )
                    signals["restrav_min_step_distance"] = restrav.get(
                        "min_step_distance"
                    )
                    signals["restrav_max_step_distance"] = restrav.get(
                        "max_step_distance"
                    )
                    signals["restrav_step_variance"] = restrav.get(
                        "step_variance"
                    )
                    signals["restrav_step_cv"] = restrav.get(
                        "step_cv"
                    )

                    signals["restrav_mean_turn_angle_deg"] = restrav.get(
                        "mean_turn_angle_deg"
                    )
                    signals["restrav_min_turn_angle_deg"] = restrav.get(
                        "min_turn_angle_deg"
                    )
                    signals["restrav_max_turn_angle_deg"] = restrav.get(
                        "max_turn_angle_deg"
                    )
                    signals["restrav_turn_angle_variance"] = restrav.get(
                        "turn_angle_variance"
                    )

                    signals["restrav_path_length"] = restrav.get(
                        "path_length"
                    )
                    signals["restrav_endpoint_distance"] = restrav.get(
                        "endpoint_distance"
                    )
                    signals["restrav_straightness_ratio"] = restrav.get(
                        "straightness_ratio"
                    )

                    # Rollout safety: no production score effect in V1.
                    signals["restrav_score_contribution"] = 0

                    log.info(
                        "RESTRAV_SHADOW: frames=%d "
                        "mean_step=%.6f mean_turn=%.3fdeg "
                        "straightness=%.6f step_cv=%.6f "
                        "contribution=0",
                        int(restrav.get("n_frames", 0)),
                        float(restrav.get("mean_step_distance", 0.0)),
                        float(restrav.get("mean_turn_angle_deg", 0.0)),
                        float(restrav.get("straightness_ratio", 0.0)),
                        float(restrav.get("step_cv", 0.0)),
                    )

                else:
                    log.info(
                        "RESTRAV_SHADOW: unavailable "
                        "(insufficient frames or geometry extraction failed)"
                    )

            except Exception as restrav_exc:
                log.debug(
                    "RESTRAV_SHADOW: skipped after error: %s",
                    restrav_exc,
                )

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
