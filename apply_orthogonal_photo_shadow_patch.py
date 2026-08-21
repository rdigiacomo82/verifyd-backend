from pathlib import Path
import sys

PHOTO = Path("photo_detection.py")
ORTHO = Path("orthogonal_photo_detector.py")

MARKER = "VERIFYD_ORTHOGONAL_PHOTO_SHADOW_V1"

if not PHOTO.exists():
    raise SystemExit("ERROR: photo_detection.py not found")

photo_text = PHOTO.read_text(encoding="utf-8")

if MARKER in photo_text or ORTHO.exists():
    raise SystemExit(
        "ERROR: Orthogonal photo patch appears to be already present. "
        "No changes made."
    )

# ============================================================
# New independent DINOv2 orthogonal photo detector
# ============================================================

ortho_code = r'''# ============================================================
# VeriFYD — orthogonal_photo_detector.py
#
# VERIFYD_ORTHOGONAL_PHOTO_SHADOW_V1
#
# DINOv2 feature-space / orthogonal decomposition analysis
# for still photographs.
#
# Research inspiration:
#   Preserve a general pretrained visual representation and
#   inspect orthogonal/residual feature structure rather than
#   relying only on generator-specific handcrafted artifacts.
#
# IMPORTANT ROLLOUT SAFETY:
#   - SHADOW MODE ONLY
#   - Does NOT alter VeriFYD photo score
#   - Does NOT alter REAL / AI / UNDETERMINED classification
#   - Does NOT alter GPT weighting
#   - contribution is always 0 in V1
#
# Reuses VeriFYD's EXISTING DINOv2 ViT-S/14 model from
# dinov2_detector.py. No additional foundation model is loaded.
# ============================================================

import os
import time
import logging
from typing import Dict

import numpy as np

log = logging.getLogger("verifyd.orthogonal_photo")


def orthogonal_photo_shadow_enabled() -> bool:
    return os.environ.get(
        "VERIFYD_ORTHO_PHOTO_SHADOW_MODE", "false"
    ).strip().lower() in ("1", "true", "yes", "on")


def _empty_result(reason: str = "") -> Dict:
    return {
        "available": False,
        "enabled": orthogonal_photo_shadow_enabled(),
        "reason": reason,

        "n_patches": 0,
        "embedding_dim": 0,

        "effective_rank": None,
        "normalized_effective_rank": None,
        "spectral_entropy": None,
        "normalized_spectral_entropy": None,

        "top1_energy_ratio": None,
        "top5_energy_ratio": None,
        "top10_energy_ratio": None,
        "top32_energy_ratio": None,
        "tail32_energy_ratio": None,

        "orthogonal_residual_ratio_r32": None,

        "cls_patch_cosine_mean": None,
        "cls_patch_cosine_std": None,

        "neighbor_cosine_mean": None,
        "neighbor_cosine_std": None,

        "patch_feature_variance": None,

        "analysis_ms": None,

        # V1 rollout protection
        "ai_score": 0,
        "contribution": 0,
        "classification_enabled": False,
    }


def _safe_float(value, digits=6):
    try:
        return round(float(value), digits)
    except Exception:
        return None


def analyze_orthogonal_photo(image_path: str) -> Dict:
    """
    Analyze a photo in DINOv2 representation space.

    This is intentionally a FEATURE COLLECTOR, not a classifier.

    Pipeline:
      1. Load image with OpenCV.
      2. Reuse VeriFYD's cached DINOv2 ViT-S/14.
      3. Extract CLS + patch token embeddings.
      4. Center patch embeddings.
      5. Compute singular spectrum.
      6. Measure effective rank, spectral concentration and
         orthogonal residual energy.
      7. Measure local neighboring-patch representation coherence.

    Returns telemetry only. contribution=0.
    """

    if not orthogonal_photo_shadow_enabled():
        return _empty_result("shadow mode disabled")

    started = time.perf_counter()

    try:
        import cv2
        import torch
        import torch.nn.functional as F
        import dinov2_detector as dino
    except Exception as exc:
        result = _empty_result(f"import failure: {exc}")
        log.warning("ORTHO_PHOTO_SHADOW: import failure: %s", exc)
        return result

    try:
        image = cv2.imread(image_path)

        if image is None:
            result = _empty_result("image could not be decoded by OpenCV")
            log.info(
                "ORTHO_PHOTO_SHADOW: unavailable image_decode_failed path=%s",
                image_path,
            )
            return result

        # Reuse existing cached DINOv2 model.
        if not dino._load_model():
            result = _empty_result("DINOv2 unavailable")
            log.info("ORTHO_PHOTO_SHADOW: unavailable DINOv2")
            return result

        model = dino._dino_model

        tensor = dino._preprocess_frame(image)
        if tensor is None:
            result = _empty_result("DINO preprocessing failed")
            return result

        with torch.no_grad():
            outputs = model(
                pixel_values=tensor,
                output_hidden_states=False,
                return_dict=True,
            )

        # HuggingFace DINOv2 ViT-S/14:
        # [batch, CLS + patches, embedding dimension]
        tokens = outputs.last_hidden_state.squeeze(0).float().cpu()

        if tokens.ndim != 2 or tokens.shape[0] < 3:
            result = _empty_result("unexpected DINO token shape")
            return result

        cls_token = tokens[0]
        patch_tokens = tokens[1:]

        n_patches = int(patch_tokens.shape[0])
        embedding_dim = int(patch_tokens.shape[1])

        # ----------------------------------------------------
        # 1. Center patch feature space
        # ----------------------------------------------------
        patch_mean = patch_tokens.mean(dim=0, keepdim=True)
        centered = patch_tokens - patch_mean

        # ----------------------------------------------------
        # 2. Singular spectrum of patch representation
        #
        # Singular values describe how many independent
        # representation directions the image uses.
        # ----------------------------------------------------
        singular_values = torch.linalg.svdvals(centered)

        singular_values = singular_values[
            torch.isfinite(singular_values)
        ]

        if singular_values.numel() < 2:
            result = _empty_result("insufficient singular spectrum")
            return result

        # S^2 gives variance / energy explained by each singular
        # direction.
        energy = singular_values.pow(2)
        total_energy = energy.sum()

        if float(total_energy.item()) <= 1e-12:
            result = _empty_result("degenerate feature spectrum")
            return result

        p = energy / total_energy
        eps = 1e-12

        # Shannon entropy of singular-energy distribution.
        spectral_entropy = -torch.sum(
            p * torch.log(p + eps)
        )

        max_entropy = torch.log(
            torch.tensor(
                float(p.numel()),
                dtype=spectral_entropy.dtype,
            )
        )

        normalized_entropy = (
            spectral_entropy / max_entropy
            if float(max_entropy.item()) > 0
            else torch.tensor(0.0)
        )

        # Effective rank = exp(entropy).
        effective_rank = torch.exp(spectral_entropy)

        normalized_effective_rank = (
            effective_rank / float(p.numel())
        )

        def energy_ratio(k: int) -> float:
            k = min(k, int(p.numel()))
            if k <= 0:
                return 0.0
            return float(p[:k].sum().item())

        top1 = energy_ratio(1)
        top5 = energy_ratio(5)
        top10 = energy_ratio(10)

        rank_k = min(32, int(p.numel()))
        top32 = energy_ratio(rank_k)
        tail32 = max(0.0, 1.0 - top32)

        # ----------------------------------------------------
        # 3. Orthogonal residual measurement
        #
        # The leading rank-32 directions form the image's
        # dominant feature subspace. Energy orthogonal to that
        # dominant subspace is the residual.
        #
        # In V1 this is SELF-referenced. Later, after collecting
        # known-real photos, we can replace/augment this with a
        # real-photo reference subspace.
        # ----------------------------------------------------
        orthogonal_residual_ratio = tail32

        # ----------------------------------------------------
        # 4. CLS / patch relationship
        # ----------------------------------------------------
        cls_norm = F.normalize(
            cls_token.unsqueeze(0),
            dim=1,
        )

        patch_norm = F.normalize(
            patch_tokens,
            dim=1,
        )

        cls_patch_cos = (
            patch_norm * cls_norm
        ).sum(dim=1)

        cls_patch_mean = float(
            cls_patch_cos.mean().item()
        )

        cls_patch_std = float(
            cls_patch_cos.std(unbiased=False).item()
        )

        # ----------------------------------------------------
        # 5. Neighboring patch representation consistency
        #
        # ViT-S/14 at 224x224 normally yields 16x16 = 256
        # patches. If the patch count is square, compare
        # horizontal + vertical neighbors.
        # ----------------------------------------------------
        side = int(round(np.sqrt(n_patches)))

        neighbor_values = []

        if side * side == n_patches:
            grid = patch_norm.reshape(
                side,
                side,
                embedding_dim,
            )

            horizontal = (
                grid[:, :-1, :] *
                grid[:, 1:, :]
            ).sum(dim=-1).reshape(-1)

            vertical = (
                grid[:-1, :, :] *
                grid[1:, :, :]
            ).sum(dim=-1).reshape(-1)

            neighbor = torch.cat(
                [horizontal, vertical],
                dim=0,
            )

            if neighbor.numel() > 0:
                neighbor_values = neighbor

        if isinstance(neighbor_values, torch.Tensor):
            neighbor_mean = float(
                neighbor_values.mean().item()
            )
            neighbor_std = float(
                neighbor_values.std(
                    unbiased=False
                ).item()
            )
        else:
            neighbor_mean = None
            neighbor_std = None

        # Overall patch-space feature variance.
        patch_feature_variance = float(
            patch_tokens.var(
                dim=0,
                unbiased=False
            ).mean().item()
        )

        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000.0

        result = {
            "available": True,
            "enabled": True,
            "reason": "",

            "n_patches": n_patches,
            "embedding_dim": embedding_dim,

            "effective_rank": _safe_float(
                effective_rank.item()
            ),
            "normalized_effective_rank": _safe_float(
                normalized_effective_rank.item()
            ),

            "spectral_entropy": _safe_float(
                spectral_entropy.item()
            ),
            "normalized_spectral_entropy": _safe_float(
                normalized_entropy.item()
            ),

            "top1_energy_ratio": _safe_float(top1),
            "top5_energy_ratio": _safe_float(top5),
            "top10_energy_ratio": _safe_float(top10),
            "top32_energy_ratio": _safe_float(top32),
            "tail32_energy_ratio": _safe_float(tail32),

            "orthogonal_residual_ratio_r32": _safe_float(
                orthogonal_residual_ratio
            ),

            "cls_patch_cosine_mean": _safe_float(
                cls_patch_mean
            ),
            "cls_patch_cosine_std": _safe_float(
                cls_patch_std
            ),

            "neighbor_cosine_mean": _safe_float(
                neighbor_mean
            ),
            "neighbor_cosine_std": _safe_float(
                neighbor_std
            ),

            "patch_feature_variance": _safe_float(
                patch_feature_variance
            ),

            "analysis_ms": _safe_float(
                elapsed_ms,
                digits=2,
            ),

            # V1 safety — telemetry only.
            "ai_score": 0,
            "contribution": 0,
            "classification_enabled": False,
        }

        log.info(
            "ORTHO_PHOTO_SHADOW: "
            "patches=%d dim=%d "
            "eff_rank=%.3f norm_rank=%.4f "
            "spec_ent=%.4f "
            "top1=%.4f top5=%.4f top10=%.4f "
            "residual32=%.4f "
            "cls_patch=%.4f "
            "neighbor=%.4f "
            "patch_var=%.4f "
            "analysis_ms=%.1f "
            "contribution=0",
            n_patches,
            embedding_dim,
            float(result["effective_rank"] or 0),
            float(result["normalized_effective_rank"] or 0),
            float(result["normalized_spectral_entropy"] or 0),
            float(result["top1_energy_ratio"] or 0),
            float(result["top5_energy_ratio"] or 0),
            float(result["top10_energy_ratio"] or 0),
            float(result["orthogonal_residual_ratio_r32"] or 0),
            float(result["cls_patch_cosine_mean"] or 0),
            float(result["neighbor_cosine_mean"] or 0),
            float(result["patch_feature_variance"] or 0),
            float(result["analysis_ms"] or 0),
        )

        return result

    except Exception as exc:
        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000.0

        result = _empty_result(str(exc))
        result["analysis_ms"] = _safe_float(
            elapsed_ms,
            digits=2,
        )

        log.warning(
            "ORTHO_PHOTO_SHADOW: analysis failed: %s",
            exc,
        )

        return result
'''

ORTHO.write_text(ortho_code, encoding="utf-8")

# ============================================================
# Patch photo_detection.py
# ============================================================

# 1. Insert shadow call immediately after signal detector.
anchor1 = '''    signal_score, signal_context = detect_ai_photo(image_path)
    log.info("Photo signal score: %d  content_type: %s",
             signal_score, signal_context.get("content_type", "photo"))

    # ── Engine 2: GPT-4o vision ──────────────────────────────
'''

replacement1 = '''    signal_score, signal_context = detect_ai_photo(image_path)
    log.info("Photo signal score: %d  content_type: %s",
             signal_score, signal_context.get("content_type", "photo"))

    # ========================================================
    # VERIFYD_ORTHOGONAL_PHOTO_SHADOW_V1
    #
    # DINOv2 representation-space forensic telemetry.
    # Shadow mode only: this result MUST NOT alter combined,
    # authenticity, label, thresholds, GPT weighting or
    # certificate behavior.
    # ========================================================
    ortho_photo = {
        "available": False,
        "enabled": False,
        "ai_score": 0,
        "contribution": 0,
        "classification_enabled": False,
    }

    try:
        from orthogonal_photo_detector import (
            analyze_orthogonal_photo,
            orthogonal_photo_shadow_enabled,
        )

        if orthogonal_photo_shadow_enabled():
            ortho_photo = analyze_orthogonal_photo(
                image_path
            )

    except Exception as _ortho_exc:
        log.warning(
            "ORTHO_PHOTO_SHADOW: skipped after error: %s",
            _ortho_exc,
        )

    # ── Engine 2: GPT-4o vision ──────────────────────────────
'''

if anchor1 not in photo_text:
    ORTHO.unlink(missing_ok=True)
    raise SystemExit(
        "ERROR: signal/GPT insertion anchor not found"
    )

photo_text = photo_text.replace(
    anchor1,
    replacement1,
    1,
)

# 2. Add shadow telemetry to detail dict.
anchor2 = '''        "threshold_real":     THRESHOLD_REAL,
        "threshold_undet":    THRESHOLD_UNDETERMINED,
    }
'''

replacement2 = '''        "threshold_real":     THRESHOLD_REAL,
        "threshold_undet":    THRESHOLD_UNDETERMINED,

        # VERIFYD_ORTHOGONAL_PHOTO_SHADOW_V1
        # Observability only. Never used in V1 scoring.
        "orthogonal_photo":   ortho_photo,
        "orthogonal_photo_available": ortho_photo.get(
            "available", False
        ),
        "orthogonal_photo_contribution": 0,
    }
'''

if anchor2 not in photo_text:
    ORTHO.unlink(missing_ok=True)
    raise SystemExit(
        "ERROR: final detail dictionary anchor not found"
    )

photo_text = photo_text.replace(
    anchor2,
    replacement2,
    1,
)

# ============================================================
# Final validation
# ============================================================

if MARKER not in photo_text:
    ORTHO.unlink(missing_ok=True)
    raise SystemExit(
        "ERROR: marker missing after patch"
    )

backup = Path(
    "photo_detection.py.before_orthogonal_photo_shadow"
)

if not backup.exists():
    backup.write_text(
        PHOTO.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print(f"Backup created: {backup}")

PHOTO.write_text(
    photo_text,
    encoding="utf-8",
)

print("")
print("SUCCESS: DINOv2 orthogonal photo shadow detector installed.")
print("")
print("Created:")
print("  orthogonal_photo_detector.py")
print("")
print("Modified:")
print("  photo_detection.py")
print("")
print("Production safeguards:")
print("  - contribution = 0")
print("  - current photo signal score unchanged")
print("  - GPT weighting unchanged")
print("  - REAL/AI/UNDETERMINED thresholds unchanged")
print("  - certificate behavior unchanged")
print("  - existing DINOv2 model reused")
print("  - no new dependencies")
print("")
print("Enable after healthy deploy with:")
print("  VERIFYD_ORTHO_PHOTO_SHADOW_MODE=true")