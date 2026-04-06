# detection.py
# Heuristic feature extraction using OpenCV (for demo purposes)
# This is NOT a reliable fake currency detector. Replace with a trained model for real use.

import numpy as np
import cv2

def _clamp(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))

def analyze_note(img_rgb):
    # Ensure correct format
    if img_rgb is None or img_rgb.size == 0:
        raise ValueError("Empty image")

    # Convert to BGR for OpenCV ops and to grayscale
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1) Sharpness via variance of Laplacian (focus measure)
    var_laplace = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2) Edge density via Canny
    v = np.median(gray)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, lower, upper)
    edge_density = float(np.sum(edges > 0) / edges.size)

    # 3) Brightness/contrast proxy: mean gray value
    mean_gray = float(np.mean(gray))

    # Normalize to heuristic sub-scores in [0,1]
    # These ranges are empirical for demo visuals
    s_sharp = _clamp(var_laplace / 1500.0)               # good if >= 1500
    s_edge  = _clamp(1.0 - abs(edge_density - 0.08) / 0.08)  # best near ~0.08
    s_bright = _clamp(1.0 - abs(mean_gray - 130.0) / 130.0)  # best near ~130

    # Weighted score (tweakable)
    score = 0.45 * s_sharp + 0.40 * s_edge + 0.15 * s_bright

    # Map to label + confidence
    label = "Likely Genuine" if score >= 0.5 else "Potentially Fake"
    confidence = 50.0 + (score - 0.5) * 100.0  # maps 0.0..1.0 to 0..100 centered at 50
    confidence = float(max(0.0, min(100.0, confidence)))

    return {
        "label": label,
        "confidence": confidence,
        "score": float(score),
        "features": {
            "var_laplace": float(var_laplace),
            "edge_density": float(edge_density),
            "mean_gray": float(mean_gray),
        }
    }
