import cv2
import numpy as np

def run_detection(video_path: str):
    """
    Returns:
        authenticity_score (0-100)
        label
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 50, "UNDETERMINED"

    noise_vals = []
    edge_vals = []
    motion_vals = []
    temporal_vals = []

    prev_gray = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 4 != 0:
            continue
        if frame_count > 240:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- SENSOR NOISE ---
        noise = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_vals.append(noise)

        # --- EDGE DETAIL ---
        edges = cv2.Canny(gray, 50, 150)
        edge_vals.append(np.mean(edges))

        # --- MOTION ---
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_vals.append(np.mean(diff))

        # --- TEMPORAL VARIATION ---
        if prev_gray is not None:
            hist1 = cv2.calcHist([gray],[0],None,[64],[0,256])
            hist2 = cv2.calcHist([prev_gray],[0],None,[64],[0,256])
            temporal_vals.append(np.mean(np.abs(hist1-hist2)))

        prev_gray = gray

    cap.release()

    if not noise_vals:
        return 50, "UNDETERMINED"

    avg_noise = np.mean(noise_vals)
    avg_edge = np.mean(edge_vals)
    avg_motion = np.mean(motion_vals) if motion_vals else 0
    temporal = np.mean(temporal_vals) if temporal_vals else 0

    # -------------------------------------------------
    # AI LIKELIHOOD SCORING
    # -------------------------------------------------
    ai_score = 50

    # AI videos usually too smooth
    if avg_noise < 120:
        ai_score += 25
    else:
        ai_score -= 10

    # AI edges too clean
    if avg_edge < 12:
        ai_score += 20
    else:
        ai_score -= 10

    # Motion too consistent
    if avg_motion < 2:
        ai_score += 15
    else:
        ai_score -= 10

    # Temporal too stable
    if temporal < 2:
        ai_score += 20
    else:
        ai_score -= 10

    ai_score = max(0, min(100, ai_score))

    # Convert to authenticity
    authenticity = 100 - ai_score

    if authenticity >= 85:
        label = "REAL"
    elif authenticity >= 60:
        label = "UNDETERMINED"
    else:
        label = "AI"

    print("AI SCORE:", ai_score, "AUTH:", authenticity)

    return authenticity, label





