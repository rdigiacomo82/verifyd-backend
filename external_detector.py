import cv2
import numpy as np

def external_ai_score(video_path):
    """
    Hybrid detector placeholder.
    Will later connect to real forensic model.
    """

    cap = cv2.VideoCapture(video_path)

    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % 30 == 0:
            frames.append(frame)

        if len(frames) >= 6:
            break

    cap.release()

    if not frames:
        return 50

    scores = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        noise = np.std(gray)
        scores.append(noise)

    avg = np.mean(scores)

    if avg < 8:
        return 80   # likely AI
    if avg < 15:
        return 50
    return 10       # likely real
