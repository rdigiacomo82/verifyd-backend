import cv2
import numpy as np
import torch
import timm

# Load pretrained vision model
model = timm.create_model("efficientnet_b0", pretrained=True)
model.eval()

def analyze_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img).unsqueeze(0)

    with torch.no_grad():
        feat = model.forward_features(img)
        score = float(torch.mean(feat))

    return score

def detect_ai(video_path):

    cap = cv2.VideoCapture(video_path)

    frame_scores = []
    noise_scores = []
    motion_scores = []

    prev_gray = None
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count > 180:
            break

        if count % 5 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # noise analysis
        noise_scores.append(np.std(gray))

        # model analysis
        frame_scores.append(analyze_frame(frame))

        # motion
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_scores.append(np.mean(diff))

        prev_gray = gray

    cap.release()

    if not frame_scores:
        return 100

    avg_model = np.mean(frame_scores)
    avg_noise = np.mean(noise_scores)
    avg_motion = np.mean(motion_scores) if motion_scores else 0

    ai_score = 100

    if avg_model < 0.2:
        ai_score -= 40

    if avg_noise < 10:
        ai_score -= 30

    if avg_motion < 2:
        ai_score -= 30

    return int(max(ai_score, 0))
