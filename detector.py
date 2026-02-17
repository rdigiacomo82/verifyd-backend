import cv2
import numpy as np

def detect_ai(video_path: str) -> int:
    """
    Returns AI likelihood score (0–100)
    HIGH = likely AI
    LOW = likely real
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 50

    noise_vals = []
    motion_vals = []
    edge_vals = []
    freq_vals = []

    prev_gray = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # sample every 4th frame
        if frame_count % 4 != 0:
            continue
        if frame_count > 240:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -----------------------
        # NOISE (real cameras = more noise)
        # -----------------------
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_vals.append(lap)

        # -----------------------
        # EDGES
        # -----------------------
        edges = cv2.Canny(gray, 50, 150)
        edge_vals.append(np.mean(edges))

        # -----------------------
        # FREQUENCY
        # -----------------------
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)

        rows, cols = gray.shape
        center = mag[rows//4:3*rows//4, cols//4:3*cols//4]
        freq_ratio = np.sum(center) / (np.sum(mag) + 1e-9)
        freq_vals.append(freq_ratio)

        # -----------------------
        # MOTION
        # -----------------------
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
    avg_freq = np.mean(freq_vals)

    # --------------------------------
    # START AI SCORE
    # --------------------------------
    ai_score = 50

    # --- NOISE ---
    if avg_noise < 80:
        ai_score += 25
    elif avg_noise > 250:
        ai_score -= 20

    # --- MOTION ---
    if avg_motion < 1.5:
        ai_score += 20
    elif avg_motion > 6:
        ai_score -= 15

    # --- EDGES ---
    if avg_edge < 12:
        ai_score += 15
    elif avg_edge > 35:
        ai_score -= 10

    # --- FREQUENCY ---
    if avg_freq < 0.55:
        ai_score += 15
    elif avg_freq > 0.75:
        ai_score -= 10

    ai_score = max(0, min(100, int(ai_score)))
    return ai_score


