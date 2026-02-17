import cv2
import numpy as np

# ---------------------------------------------------
# CORE DETECTOR
# ---------------------------------------------------

def run_detection(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return 50, "UNDETERMINED"

    noise_vals = []
    motion_vals = []
    edge_vals = []
    entropy_vals = []

    prev_gray = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # sample every 5th frame, max 120 frames
        if frame_count % 5 != 0:
            continue
        if frame_count > 600:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---------------------------------------------------
        # 1. SENSOR NOISE (REAL CAMERAS HAVE MORE)
        # ---------------------------------------------------
        noise = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_vals.append(noise)

        # ---------------------------------------------------
        # 2. EDGE COMPLEXITY
        # ---------------------------------------------------
        edges = cv2.Canny(gray, 50, 150)
        edge_vals.append(np.mean(edges))

        # ---------------------------------------------------
        # 3. ENTROPY (AI tends to be too uniform)
        # ---------------------------------------------------
        hist = cv2.calcHist([gray], [0], None, [256], [0,256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        entropy_vals.append(entropy)

        # ---------------------------------------------------
        # 4. MOTION INSTABILITY (real cameras shake slightly)
        # ---------------------------------------------------
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_vals.append(np.mean(diff))

        prev_gray = gray

    cap.release()

    if not noise_vals:
        return 50, "UNDETERMINED"

    avg_noise = np.mean(noise_vals)
    avg_motion = np.mean(motion_vals) if motion_vals else 0
    avg_edges = np.mean(edge_vals)
    avg_entropy = np.mean(entropy_vals)

    # ---------------------------------------------------
    # SCORING (AI likelihood)
    # start at 50
    # ---------------------------------------------------
    ai_score = 50

    # ---------- NOISE ----------
    if avg_noise < 80:       # too clean → AI
        ai_score += 25
    elif avg_noise > 300:    # real camera
        ai_score -= 20

    # ---------- MOTION ----------
    if avg_motion < 1.5:     # too stable → AI
        ai_score += 20
    elif avg_motion > 5:
        ai_score -= 10

    # ---------- EDGES ----------
    if avg_edges < 5:        # too smooth
        ai_score += 15
    elif avg_edges > 25:
        ai_score -= 10

    # ---------- ENTROPY ----------
    if avg_entropy < 5.0:
        ai_score += 15
    elif avg_entropy > 6.5:
        ai_score -= 10

    ai_score = max(0, min(100, int(ai_score)))

    # ---------------------------------------------------
    # FINAL CLASSIFICATION
    # ---------------------------------------------------
    if ai_score >= 70:
        return ai_score, "AI"
    elif ai_score >= 45:
        return ai_score, "UNDETERMINED"
    else:
        return ai_score, "REAL"


