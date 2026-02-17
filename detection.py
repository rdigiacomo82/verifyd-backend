import cv2
import numpy as np
import random

# ----------------------------------------------------
# Core AI detection tuned for:
# phone video vs AI social clips
# ----------------------------------------------------

def run_detection(video_path):
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

        # analyze only first ~10 seconds
        if frame_count > 300:
            break

        if frame_count % 5 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- noise ---
        noise = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_vals.append(noise)

        # --- brightness variation ---
        brightness_vals.append(np.mean(gray))

        # --- edges ---
        edges = cv2.Canny(gray, 50, 150)
        edge_vals.append(np.mean(edges))

        # --- motion ---
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_vals.append(np.mean(diff))

        prev_gray = gray

    cap.release()

    if not noise_vals:
        return 50, "UNDETERMINED"

    avg_noise = np.mean(noise_vals)
    avg_motion = np.mean(motion_vals) if motion_vals else 0
    motion_var = np.var(motion_vals) if len(motion_vals) > 1 else 0
    brightness_var = np.var(brightness_vals)
    avg_edges = np.mean(edge_vals)

    # ----------------------------------------------------
    # AI likelihood score (0–100 where HIGH = AI)
    # ----------------------------------------------------
    ai_score = 50

    # --- SENSOR NOISE ---
    if avg_noise < 80:
        ai_score += 25   # too clean = AI
    elif avg_noise > 300:
        ai_score -= 20   # real camera noise

    # --- MOTION ---
    if avg_motion < 1.5:
        ai_score += 20   # too stable
    elif avg_motion > 6:
        ai_score -= 15   # real handheld motion

    # motion variance
    if motion_var < 0.5:
        ai_score += 15

    # --- BRIGHTNESS variation ---
    if brightness_var < 10:
        ai_score += 15   # lighting too perfect
    elif brightness_var > 40:
        ai_score -= 10

    # --- EDGE DETAIL ---
    if avg_edges < 8:
        ai_score += 10
    elif avg_edges > 25:
        ai_score -= 10

    # clamp
    ai_score = max(0, min(100, ai_score))

    # small randomness to avoid identical results
    ai_score += random.randint(-3, 3)
    ai_score = max(0, min(100, ai_score))

    # ----------------------------------------------------
    # Convert to REAL score
    # ----------------------------------------------------
    real_score = 100 - ai_score

    if real_score >= 65:
        return real_score, "REAL"
    elif real_score >= 40:
        return real_score, "UNDETERMINED"
    else:
        return real_score, "AI"









