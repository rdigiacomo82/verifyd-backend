import cv2
import numpy as np

# ============================================================
# VeriFYD Advanced Detector v1
# Returns AUTHENTICITY score (0–100)
# 100 = real camera
# 0 = AI generated
# ============================================================

def run_detection(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return 50, "UNDETERMINED"

    frame_count = 0
    prev_gray = None

    motion_vals = []
    jitter_vals = []
    detail_vals = []
    brightness_vals = []

    # only analyze first ~10 seconds (≈300 frames max)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count > 300:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- detail (real cameras have more high-freq noise)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        detail_vals.append(lap)

        # --- brightness variation (real cameras fluctuate)
        brightness_vals.append(np.mean(gray))

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_vals.append(np.mean(diff))

            # micro jitter (phone shake)
            shift = np.sum(diff) / (diff.shape[0] * diff.shape[1])
            jitter_vals.append(shift)

        prev_gray = gray

    cap.release()

    if not motion_vals:
        return 50, "UNDETERMINED"

    avg_motion = np.mean(motion_vals)
    motion_var = np.var(motion_vals)
    avg_detail = np.mean(detail_vals)
    brightness_var = np.var(brightness_vals)
    jitter = np.mean(jitter_vals)

    # ============================================================
    # AUTHENTICITY SCORE (start neutral at 50)
    # ============================================================

    score = 50

    # ---- real camera motion randomness
    if motion_var > 5:
        score += 15
    else:
        score -= 15

    # ---- micro camera shake (very strong real signal)
    if jitter > 2:
        score += 15
    else:
        score -= 15

    # ---- real cameras have natural detail/noise
    if avg_detail > 200:
        score += 10
    else:
        score -= 10

    # ---- brightness fluctuation
    if brightness_var > 2:
        score += 10
    else:
        score -= 10

    # ---- extremely smooth motion → AI
    if avg_motion < 1:
        score -= 15

    # clamp
    score = int(max(0, min(100, score)))

    # ============================================================
    # CLASSIFICATION
    # ============================================================

    if score >= 85:
        return score, "REAL"
    elif score >= 60:
        return score, "UNDETERMINED"
    else:
        return score, "AI"










