import cv2
import numpy as np

def run_detection(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return 50, "UNDETERMINED"

    noise_vals = []
    motion_vals = []
    edge_vals = []
    entropy_vals = []

    prev_gray = None
    frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames += 1

        # sample frames
        if frames % 6 != 0:
            continue
        if frames > 300:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---------- noise ----------
        noise = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_vals.append(noise)

        # ---------- edges ----------
        edges = cv2.Canny(gray, 80, 160)
        edge_vals.append(np.mean(edges))

        # ---------- entropy ----------
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        hist = hist / (hist.sum()+1e-6)
        entropy = -np.sum(hist * np.log2(hist + 1e-9))
        entropy_vals.append(entropy)

        # ---------- motion ----------
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_vals.append(np.mean(diff))

        prev_gray = gray

    cap.release()

    if not noise_vals:
        return 50, "UNDETERMINED"

    avg_noise = np.mean(noise_vals)
    avg_edges = np.mean(edge_vals)
    avg_entropy = np.mean(entropy_vals)
    avg_motion = np.mean(motion_vals) if motion_vals else 0

    # ----------------------------------------
    # START SCORE (AI likelihood)
    # ----------------------------------------
    ai_score = 50

    # ---- TOO CLEAN → AI ----
    if avg_noise < 60:
        ai_score += 25
    elif avg_noise > 200:
        ai_score -= 10

    # ---- TOO SMOOTH MOTION ----
    if avg_motion < 1.2:
        ai_score += 20
    elif avg_motion > 4:
        ai_score -= 10

    # ---- PERFECT EDGES ----
    if avg_edges < 4:
        ai_score += 15
    elif avg_edges > 20:
        ai_score -= 5

    # ---- LOW ENTROPY ----
    if avg_entropy < 5.2:
        ai_score += 20
    elif avg_entropy > 6.2:
        ai_score -= 5

    ai_score = int(max(0, min(100, ai_score)))

    # ----------------------------------------
    # CLASSIFY
    # ----------------------------------------
    if ai_score >= 80:
        return ai_score, "AI"
    elif ai_score >= 60:
        return ai_score, "UNDETERMINED"
    else:
        return ai_score, "REAL"



