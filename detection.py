import cv2
import numpy as np

def run_detection(video_path: str):
    """
    Returns REAL score 0–100
    HIGH = real
    LOW = AI
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 50, "Undetermined"

    noise_vals = []
    motion_vals = []
    exposure_vals = []

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

        # sensor noise
        noise = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_vals.append(noise)

        # exposure variation
        exposure_vals.append(np.std(gray))

        # motion
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_vals.append(np.mean(diff))

        prev_gray = gray

    cap.release()

    if not noise_vals:
        return 50, "Undetermined"

    avg_noise = np.mean(noise_vals)
    avg_motion = np.mean(motion_vals) if motion_vals else 0
    avg_exposure = np.mean(exposure_vals)

    real_score = 50

    # real cameras = noise
    if avg_noise > 150:
        real_score += 25
    elif avg_noise < 60:
        real_score -= 25

    # real cameras = motion
    if avg_motion > 3:
        real_score += 25
    elif avg_motion < 1:
        real_score -= 25

    # real cameras = exposure variation
    if avg_exposure > 25:
        real_score += 20
    elif avg_exposure < 12:
        real_score -= 20

    real_score = max(0, min(100, int(real_score)))

    if real_score >= 85:
        label = "REAL"
    elif real_score >= 60:
        label = "Undetermined"
    else:
        label = "AI"

    return real_score, label




