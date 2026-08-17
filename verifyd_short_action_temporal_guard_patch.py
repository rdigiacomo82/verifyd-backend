from pathlib import Path
import shutil
import time
import py_compile

TARGET = Path("detection.py")

if not TARGET.exists():
    raise SystemExit("ERROR: detection.py not found. Run this from the VeriFYD repo root.")

text = TARGET.read_text(encoding="utf-8")
original = text

PATCH_MARKER = "VERIFYD_SHORT_ACTION_TEMPORAL_GUARD_V1"

if PATCH_MARKER in text:
    raise SystemExit("ERROR: Patch marker already present. Aborting to avoid double patching.")

backup = Path(f"detection.py.before_short_action_temporal_guard_{int(time.time())}.bak")
shutil.copy2(TARGET, backup)
print(f"Backup created: {backup}")

# -------------------------------------------------------------------
# PATCH 1
# DeepfakeDetector is face-oriented. On action/cinematic footage a low
# deepfake score must not be treated as proof that the whole scene is real.
# Preserve positive AI contributions, neutralize negative contributions.
# -------------------------------------------------------------------

old_df = '''                    df_contribution = get_deepfake_contribution(df_score, _score, _ct)
                    _ctx["deepfake_contribution"] = df_contribution
                    if df_contribution != 0:
                        log.info(
                            "DeepfakeDetector @%.0f%%: score=%d contribution=%+d "
                            "(signal=%d content=%s skin=%.3f)",
                            _pct * 100, df_score, df_contribution,
                            _score, _ct, _skin,
                        )
                        _score = int(round(min(100, max(0, _score + df_contribution))))
'''

new_df = '''                    df_contribution = get_deepfake_contribution(df_score, _score, _ct)

                    # VERIFYD_SHORT_ACTION_TEMPORAL_GUARD_V1
                    # Face-oriented deepfake models can score a fully synthetic
                    # action scene as "real" when the rendered face itself is clean.
                    # For action/cinematic footage, low DeepfakeDetector scores are
                    # therefore neutral evidence rather than proof of camera origin.
                    if _ct in ("action", "cinematic") and df_contribution < 0:
                        log.info(
                            "DeepfakeDetector @%.0f%%: negative contribution %+d suppressed "
                            "for %s whole-scene generative risk",
                            _pct * 100, df_contribution, _ct,
                        )
                        df_contribution = 0

                    _ctx["deepfake_contribution"] = df_contribution
                    if df_contribution != 0:
                        log.info(
                            "DeepfakeDetector @%.0f%%: score=%d contribution=%+d "
                            "(signal=%d content=%s skin=%.3f)",
                            _pct * 100, df_score, df_contribution,
                            _score, _ct, _skin,
                        )
                        _score = int(round(min(100, max(0, _score + df_contribution))))
'''

if old_df not in text:
    raise SystemExit(
        "ERROR: Deepfake contribution anchor not found exactly. "
        "No changes written."
    )

text = text.replace(old_df, new_df, 1)

# -------------------------------------------------------------------
# PATCH 2
# Add a narrow short portrait action temporal guard AFTER normal fusion
# and BEFORE the existing device-recorded override.
#
# This intentionally requires several signals to agree:
#   action/cinematic
#   single canonical clip (<12s strategy in video.py)
#   portrait framing
#   extreme motion lockstep
#   extreme temporal color variance
#   high inter-channel correlation
#
# This exact composite matches the missed AI clip while avoiding a broad
# TCV-only or channel-correlation-only override.
# -------------------------------------------------------------------

anchor = '''    # ── Device-recorded / screen-recorded AI override ─────────────
'''

insert = '''    # ── Short portrait action temporal AI guard ──────────────────
    # VERIFYD_SHORT_ACTION_TEMPORAL_GUARD_V1
    #
    # Modern generative video can look convincingly photographic frame-by-frame,
    # especially after social-media recompression or VeriFYD canonicalization.
    # In short vertical action clips, require a narrow multi-signal temporal
    # composite before overriding a low GPT / low blended score.
    #
    # video.py uses one canonical clip for videos under 12 seconds, so n_clips==1
    # is used here as a conservative short-form proxy.
    def _sat_float(_ctx, *keys, default=0.0):
        for _key in keys:
            try:
                _val = _ctx.get(_key)
                if _val is not None:
                    return float(_val)
            except Exception:
                pass
        return float(default)

    _short_action_ctxs = all_signal_contexts or []
    _short_action_trigger = False
    _short_action_details = {}

    if n_clips == 1 and content_type in ("action", "cinematic") and _short_action_ctxs:
        _sa_ctx = _short_action_ctxs[0]

        _sa_portrait = bool(_sa_ctx.get("portrait", False))
        _sa_sync = _sat_float(_sa_ctx, "motion_sync", default=1.0)
        _sa_tcv = _sat_float(_sa_ctx, "tcv", default=0.0)
        _sa_chan = _sat_float(
            _sa_ctx,
            "channel_corr",
            "chan_corr",
            "channel_correlation",
            "inter_channel_corr",
            default=0.0,
        )
        _sa_motion = _sat_float(_sa_ctx, "motion", "avg_motion", default=0.0)

        # Production miss 581d8cb2...:
        # action=True, portrait=True, sync=.052, TCV=2470,
        # channel correlation=.8987, motion=23.7.
        #
        # Thresholds intentionally leave margin around that case while still
        # requiring all independent measurements to agree.
        _short_action_trigger = (
            _sa_portrait
            and _sa_motion >= 15.0
            and _sa_sync <= 0.065
            and _sa_tcv >= 1500.0
            and _sa_chan >= 0.885
        )

        _short_action_details = {
            "portrait": _sa_portrait,
            "motion": _sa_motion,
            "motion_sync": _sa_sync,
            "tcv": _sa_tcv,
            "channel_corr": _sa_chan,
        }

    if _short_action_trigger:
        old_combined = combined_ai_score

        # Do not allow the ordinary both-real fusion path to certify a clip
        # when an independent, high-risk temporal composite has fired.
        combined_ai_score = max(combined_ai_score, 55.0)
        mode = "short-action-temporal-guard"

        log.warning(
            "SHORT_ACTION_TEMPORAL_GUARD: portrait=%s motion=%.2f "
            "motion_sync=%.3f tcv=%.2f chan_corr=%.4f "
            "gpt=%d signal=%d combined %.1f→%.1f",
            _short_action_details.get("portrait"),
            _short_action_details.get("motion", 0.0),
            _short_action_details.get("motion_sync", 1.0),
            _short_action_details.get("tcv", 0.0),
            _short_action_details.get("channel_corr", 0.0),
            gpt_ai_score,
            signal_ai_score,
            old_combined,
            combined_ai_score,
        )

'''

if anchor not in text:
    raise SystemExit(
        "ERROR: Device override anchor not found. "
        "No changes written."
    )

text = text.replace(anchor, insert + anchor, 1)

if text == original:
    raise SystemExit("ERROR: Nothing changed.")

TARGET.write_text(text, encoding="utf-8")

try:
    py_compile.compile(str(TARGET), doraise=True)
except Exception:
    shutil.copy2(backup, TARGET)
    raise SystemExit(
        f"ERROR: detection.py compile failed. Restored backup: {backup}"
    )

print()
print("SUCCESS: verifyd_short_action_temporal_guard_patch applied.")
print("Changed file: detection.py")
print("Patch marker:", PATCH_MARKER)
print("Compile check: PASS")
print()
print("Next run:")
print("  git diff -- detection.py")
print("  py -m py_compile detection.py detector.py gpt_vision.py video.py worker.py queue_helper.py")