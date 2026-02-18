# ============================================================
#  VeriFYD — external_detector.py
#
#  External / third-party detector interface.
#
#  Previously a standalone stub with its own scoring logic.
#  Now delegates to the main detection pipeline so there is
#  exactly ONE engine across the entire project.
#
#  If you later integrate a third-party forensic API or a
#  PyTorch model, replace the body of external_ai_score()
#  here without touching detection.py or detector.py.
# ============================================================

from detection import run_detection


def external_ai_score(video_path: str) -> int:
    """
    Returns an AI score 0–100 (HIGH = likely AI) for the given video.

    Currently delegates to the main VeriFYD detection engine.
    Replace the body of this function to integrate a third-party
    forensic model or API without affecting the rest of the system.
    """
    authenticity, label, detail = run_detection(video_path)
    return detail["ai_score"]
