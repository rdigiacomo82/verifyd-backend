import cv2
import numpy as np
import torch
import torchvision.transforms as T

device = "cpu"

# Load synthetic image detector backbone
model = torch.hub.load(
    "pytorch/vision:v0.10.0",
    "resnet18",
    pretrained=True
)

model.eval()
model.to(device)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224,224)),
    T.ToTensor()
])

def analyze_frame(frame):
    img = transform(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(img)
        score = float(torch.mean(feat))

    return score

def detect_ai(video_path):

    cap = cv2.VideoCapture(video_path)

    scores = []
    noise_scores = []
    motion_scores = []

    prev = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 5 != 0:
            continue

        if frame_count > 300:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # visual model score
        scores.append(analyze_frame(frame))

        # noise check
        noise_scores.append(np.std(gray))

        # motion consistency
        if prev is not None:
            diff = cv2.absdiff(gray, prev)
            motion_scores.append(np.mean(diff))

        prev = gray

    cap.release()

    if not scores:
        return 50

    avg_model = np.mean(scores)
    avg_noise = np.mean(noise_scores)
    avg_motion = np.mean(motion_scores) if motion_scores else 0

    ai_score = 100

    if avg_model < 0.05:
        ai_score -= 40

    if avg_noise < 12:
        ai_score -= 30

    if avg_motion < 2:
        ai_score -= 30

    return int(max(ai_score,0))


