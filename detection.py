import cv2
import numpy as np

# ============================================================
# VeriFYD Detection Engine v2 (Balanced + Stable)
# ============================================================

def run_detection(video_path):
    """
    Returns:
        authenticity_score (0–100)
        label: REAL / UNDETERMINED / AI
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        # If video can't be read, assume uncertain
        return 50, "UNDETERMINED"

    frame_count = 0
    noise_vals = []
    motion_vals = []
    edge_vals = []

    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Only analyze first ~10 seconds worth of frames
        if frame_count > 300:
            break

        if frame_count % 5 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -------------------------
        # SENSOR NOISE
        # -------------------------
        noise = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_vals.append(noise)

        # -------------------------
        # EDGE DETAIL
        # -------------------------
        edges = cv2.Canny(gray, 50, 150)
        edge_vals.append(np.mean(edges))

        # -------------------------
        # MOTION
        # -------------------------
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_vals.append(np.mean(diff))

        prev_gray = gray

    cap.release()

    if not noise_vals:
        return 50, "UNDETERMINED"

    avg_noise = np.mean(noise_vals)
    avg_edges = np.mean(edge_vals)
    avg_motion = np.mean(motion_vals) if motion_vals else 0

    # =========================================================
    # AI LIKELIHOOD SCORE (higher = more likely AI)
    # =========================================================
    ai_score = 50

    # Low sensor noise → AI
    if avg_noise < 120:
        ai_score += 25
    elif avg_noise > 400:
        ai_score -= 15

    # Weak edges → AI
    if avg_edges < 12:
        ai_score += 15
    elif avg_edges > 35:
        ai_score -= 10

    # Too smooth motion → AI
    if avg_motion < 1.2:
        ai_score += 15
    elif avg_motion > 6:
        ai_score -= 10

    ai_score = max(0, min(100, ai_score))

    # =========================================================
    # CONVERT TO AUTHENTICITY
    # =========================================================
    authenticity = 100 - ai_score

    # Slight realism boost
    if authenticity > 70 and avg_noise > 250:
        authenticity += 5

    authenticity = max(0, min(100, authenticity))

    # =========================================================
    # FINAL LABEL
    # =========================================================
    if authenticity >= 85:
        label = "REAL"
    elif authenticity >= 60:
        label = "UNDETERMINED"
    else:
        label = "AI"

    return int(authenticity), label

