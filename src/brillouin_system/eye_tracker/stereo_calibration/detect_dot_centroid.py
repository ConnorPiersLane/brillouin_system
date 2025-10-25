import cv2
import numpy as np

def detect_dot_centroid(gray, min_area=50):
    # Binary inverse: black dot on white paper
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # Clean small noise
    bw = cv2.medianBlur(bw, 3)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    # pick largest blob
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    M = cv2.moments(c)
    cx = M["m10"]/M["m00"]; cy = M["m01"]/M["m00"]
    # subpixel refine in a small window
    pt = np.array([[cx, cy]], np.float32)
    term = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    cv2.cornerSubPix(gray, pt, (5,5), (-1,-1), term)
    return float(pt[0,0]), float(pt[0,1])
