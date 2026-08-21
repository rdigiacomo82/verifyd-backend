from pathlib import Path
import sys

path = Path("dinov2_detector.py")

if not path.exists():
    raise SystemExit("ERROR: dinov2_detector.py not found in current directory")

text = path.read_text(encoding="utf-8")

MARKER = "VERIFYD_RESTRAV_SHADOW_V1"

if MARKER in text:
    print("ReStraV shadow patch already present. No changes made.")
    sys.exit(0)

original = text

# ------------------------------------------------------------------
# 1. Configuration block
# ------------------------------------------------------------------
anchor = '_IMG_SIZE     = 224   # input resolution (14*16=224)\n'

if anchor not in text:
    raise SystemExit("ERROR: DINO configuration anchor not found")

config_block = r'''

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

'''

text = text.replace(anchor, anchor + config_block, 1)


# ------------------------------------------------------------------
# 2. ReStraV temporal-geometry helpers
# ------------------------------------------------------------------
anchor = '\ndef _extract_frames(video_path: str, max_frames: int) -> List[np.ndarray]:\n'

if anchor not in text:
    raise SystemExit("ERROR: _extract_frames anchor not found")

helper_block = r'''

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

'''

text = text.replace(anchor, "\n" + helper_block + anchor, 1)


# ------------------------------------------------------------------
# 3. Add ReStraV fields to returned DINO signals
# ------------------------------------------------------------------
old_signals = '''    signals: Dict = {
        "patch_var":       None,
        "cls_patch_sim":   None,
        "cls_consistency": None,
        "dino_score":      0,
        "available":       False,
    }
'''

new_signals = '''    signals: Dict = {
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
'''

if old_signals not in text:
    raise SystemExit("ERROR: analyze_dinov2 signals block anchor not found")

text = text.replace(old_signals, new_signals, 1)


# ------------------------------------------------------------------
# 4. Run ReStraV shadow telemetry AFTER current DINO score is settled.
#
# IMPORTANT:
# Existing score and contribution logic are untouched.
# ------------------------------------------------------------------
old_tail = '''        score = max(0, min(100, score))
        signals["dino_score"] = score

        log.info("DINOv2: score=%d [%s]", score, " ".join(components))
        return score, signals
'''

new_tail = '''        score = max(0, min(100, score))
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
'''

if old_tail not in text:
    raise SystemExit("ERROR: DINO score tail anchor not found")

text = text.replace(old_tail, new_tail, 1)


# ------------------------------------------------------------------
# Final validation
# ------------------------------------------------------------------
if text == original:
    raise SystemExit("ERROR: patch produced no changes")

if text.count(MARKER) < 3:
    raise SystemExit(
        "ERROR: patch markers incomplete; refusing to write file"
    )

backup = path.with_suffix(".py.before_restrav_shadow")

if not backup.exists():
    backup.write_text(original, encoding="utf-8")
    print(f"Backup created: {backup}")

path.write_text(text, encoding="utf-8")

print("SUCCESS: ReStraV shadow patch applied to dinov2_detector.py")
print("")
print("Production safeguards:")
print("  - current DINO score unchanged")
print("  - current DINO contribution unchanged")
print("  - final VeriFYD thresholds unchanged")
print("  - ReStraV score contribution = 0")
print("  - no new dependencies")
print("")
print("Enable later with:")
print("  VERIFYD_RESTRAV_SHADOW_MODE=true")