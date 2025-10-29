
import cv2

def detect_dot_with_blob(gray):
    h, w = gray.shape[:2]
    params = cv2.SimpleBlobDetector_Params()
    # params.filterByColor = True; params.blobColor = 0  # dark
    # params.filterByArea = True; params.minArea = 10; params.maxArea = 3000
    # params.filterByCircularity = True; params.minCircularity = 0.6
    # params.filterByConvexity = True; params.minConvexity = 0.5
    # params.filterByInertia = False

    det = cv2.SimpleBlobDetector_create(params)
    keypoints = det.detect(gray)

    # # mask out the border ring by distance from center
    # cx, cy = w/2, h/2
    # radius = min(h, w)/2 - border_margin
    # keypoints = [kp for kp in keypoints if (kp.pt[0]-cx)**2 + (kp.pt[1]-cy)**2 <= radius*radius]

    if not keypoints:
        return None
    kp = min(keypoints, key=lambda k: k.size)  # prefer the smallest
    return float(kp.pt[0]), float(kp.pt[1])

