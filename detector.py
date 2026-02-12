import cv2
import numpy as np
import torch
import torchvision.transforms as T
from skimage import filters
from skimage.metrics import structural_similarity as ssim

# ===============================
# LOAD MODEL
# ===============================

device = "cpu"

model = torch.hub.load(
    "pytorch/vision:v0.10.0",
    "mobilenet_v3_large",
    pretrained=True
)
model.eval()
model.to(device)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor()
])

# ===============================
# FRAME AI SCORE
# ===============================

def frame_ai_score(frame):
    img = transform(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model.features(img)
        score = float(torch.mean(feat))

    return score

# ===============================
# FREQUENCY ANALYSIS
# ===============================

def frequency_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    return np.mean(magnitude)

# ===============================
# TEMPORAL CONSISTENCY
# ===============================

def temporal_score(prev, curr):
    if prev is None:
        return 0
    return ssim(prev, curr)

# ===============================
# MAIN DETECTOR
# ===============================

def detect_ai(video_path):

    cap = cv2.VideoCapture(video_path)

    frame_scores = []
    freq_scores = []
    temporal_scores = []

    prev_gray = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # sample frames
        if frame_count % 5 != 0:
            continue

        if frame_count > 240:
            break

        # model score
        frame_scores.append(frame_ai_score(frame))

        # frequency
        freq_scores.append(frequency_score(frame))

        # temporal
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        temporal_scores.append(temporal_score(prev_gray, gray))
        prev_gray = gray

    cap.release()

    if not frame_scores:
        return 50

    avg_frame = np.mean(frame_scores)
    avg_freq = np.mean(freq_scores)
    avg_temp = np.mean(temporal_scores)

    # ===============================
    # SCORING
    # ===============================

    score = 100

    if avg_frame < 0.15:
        score -= 40

    if avg_freq < 3:
        score -= 30

    if avg_temp > 0.98:
        score -= 30

    return int(max(score, 0))

