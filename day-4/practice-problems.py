import cv2 
import numpy as np
from langchain.tools import tool 

# 1
@tool 
def extract_keyframes(video_path, seconds_interval=30):
    """saves exactly one frame every 30 seconds of video to disk."""
    cap = cv2.VideoCapture(video_path)
    ret, _ = cap.read()
    if not ret:
        raise FileNotFoundError("Video not found")
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds_interval)
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video ended")
            break 
        cv2.imwrite(f"outputs/frame{idx+1}.jpg", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                    break 
        
    cv2.destroyAllWindows()
    cap.release()
    return 

# 2
@tool 
def detect_camera_tampering(frame) -> bool:
      """returns True if a frame is completely pitch black (mean pixel value < 5) or completely washed out (mean > 250)."""
      return cv2.mean(frame) > 250 or cv2.mean(frame) < 5

# 3
@tool 
def frame_difference_score(img1, img2):
      """takes two consecutive frames, applies a slight Gaussian blur to both, uses cv2.absdiff(img1, img2), and returns the sum of all absolute differences."""
      g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
      g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

      gauss1 = cv2.GaussianBlur(g1, (5,5), 0)
      gauss2 = cv2.GaussianBlur(g2, (5,5), 0)

      diff = cv2.absdiff(gauss1, gauss2)
      return np.sum(diff)

# 4
def frame_rate(video: str):
    # returns duration in seconds
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return None 
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        duration = frames / fps 
    else:
        duration = 0
    return duration
        
      
