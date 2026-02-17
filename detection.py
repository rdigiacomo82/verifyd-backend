import cv2
import numpy as np

# ============================================================
# VeriFYD Detection Engine v3 (Tuned for Phone vs AI Avatar)
# ============================================================

def run_detection(video_path):
    """
    Returns:
        score (0-100)
        label ("REAL", "AI", "UNDETERMINED")
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return 50, "UNDETERMINED"

    noise_vals = []
    motion_vals = []
    brightness_vals = []
    edge_vals = []

    prev_gray = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # sample every 5th frame (first ~10 sec)
        if frame_count % 5 != 0:
            continue

        if frame_count > 300:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -----------------------
        # SENSOR NOISE
        # -----------------------
        noise = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_vals.append(noise)

        # -----------------------
        # EDGE DETAIL
        # -----------------------
        edges = cv2.Canny(gray, 50, 150)
        edge_vals.append(np.mean(edges))

        # -----------------------
        # BRIGHTNESS VARIATION
        # real cameras fluctuate
        # AI often stable
        # -----------------------
        brightness_vals.append(np.mean(gray))

        # -----------------------
        # MOTION
        # -----------------------
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_vals.append(np.mean(diff))

        prev_gray = gray

    cap.release()

    if len(noise_vals) == 0:
        return 50, "UNDETERMINED"

    avg_noise = np.mean(noise_vals)
    noise_var = np.var(noise_vals)

    avg_motion = np.mean(motion_vals) if motion_vals else 0
    motion_var = np.var(motion_vals) if len(motion_vals) > 1 else 0

    brightness_var = np.var(brightness_vals)
    avg_edges = np.mean(edge_vals)

    # ============================================================
    # REAL SCORE START
    # ============================================================
    real_score = 50

    # ------------------------------------------------------------
    # NOISE (phone cameras always noisy)
    # ------------------------------------------------------------
    if avg_noise > 300:
        real_score += 20
    elif avg_noise < 120:
        real_score -= 25

    # noise variance (real fluctuates)
    if noise_var > 200:
        real_score += 10
    else:
        real_score -= 10

    # ------------------------------------------------------------
    # MOTION
    # ------------------------------------------------------------
    if avg_motion > 5:
        real_score += 15
    elif avg_motion < 2:
        real_score -= 20

    if motion_var > 2:
        real_score += 10
    else:
        real_score -= 10

    # ------------------------------------------------------------
    # BRIGHTNESS FLUCTUATION
    # real outdoor video changes constantly
    # ------------------------------------------------------------
    if brightness_var > 20:
        real_score += 15
    else:
        real_score -= 15

    # ------------------------------------------------------------
    # EDGE DETAIL
    # AI often too smooth
    # ------------------------------------------------------------
    if avg_edges > 20:
        real_score += 10
    elif avg_edges < 8:
        real_score -= 15

    # clamp
    real_score = max(0, min(100, int(real_score)))

    # ============================================================
    # LABEL
    # ============================================================
    if real_score >= 85:
        return real_score, "REAL"
    elif real_score >= 60:
        return real_score, "UNDETERMINED"
    else:
        return real_score, "AI"






