import random

def run_detection(path):
    score = random.randint(55, 90)

    if score >= 70:
        return score, "REAL"
    elif score >= 45:
        return score, "REVIEW"
    else:
        return score, "AI"
