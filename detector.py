import cv2
import numpy as np

def detect_ai(video_path: str) -> int:
    """
    Returns AI likelihood (0–100)
    HIGH = AI
    LOW = real
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 50

    noise_vals = []
    motion_vals = []
    edge_vals = []
    exposure_vals = []

    prev_gray = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # sample every 3rd frame
        if frame_count % 3 != 0:
            continue
        if frame_count > 240:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -------------------
        # NOISE
        # -------------------
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_vals.append(lap)

        # -------------------
        # EDGES
        # -------------------
        edges = cv2.Canny(gray, 50, 150)
        edge_vals.append(np.mean(edges))

        # -------------------
        # EXPOSURE VARIATION
        # real cameras fluctuate slightly
        # -------------------
        exposure_vals.append(np.std(gray))

        # -------------------
        # MOTION
        # -------------------
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_vals.append(np.mean(diff))

        prev_gray = gray

    cap.release()

    if not noise_vals:
        return 50

    avg_noise = np.mean(noise_vals)
    avg_edge = np.mean(edge_vals)
    avg_motion = np.mean(motion_vals) if motion_vals else 0
    avg_exposure = np.mean(exposure_vals)

    # ------------------------------------
    # START AI SCORE
    # ------------------------------------
    ai_score = 50

    # ---- NOISE ----
    # real camera has noise
    if avg_noise < 60:
        ai_score += 30
    elif avg_noise > 200:
        ai_score -= 25

    # ---- MOTION ----
    if avg_motion < 1.2:
        ai_score += 25
    elif avg_motion > 5:
        ai_score -= 20

    # ---- EDGES ----
    if avg_edge < 10:
        ai_score += 20
    elif avg_edge > 28:
        ai_score -= 15

    # ---- EXPOSURE ----
    if avg_exposure < 18:
        ai_score += 15
    elif avg_exposure > 40:
        ai_score -= 10

    ai_score = max(0, min(100, int(ai_score)))
    return ai_score



