import cv2 
import numpy as np

def extract_motion_magnitude(frame1_path: str, frame2_path: str) -> float:
    f1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
    if not f1: raise FileNotFoundError("Frame 1 not found")
    f2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)
    if not f2: raise FileNotFoundError("Frame 2 not found")

    hsv = np.zeros_like(f1)
    hsv[..., 1] = 255
    flow = cv2.calcOpticalFlowFarneback(
        f1, f2, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1]) # return (H, W, 2)
    # returning the mean of magnitude
    return np.mean(mag)
    
    
