# HANDSON 1
import cv2 

def detect_motion_events(video_path: str) -> list:
    cap = cv2.VideoCapture(video_path)
    backsub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=20, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    idx = []
    for i in range(100):
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Unable to fetch video frames")
        fg_mask = backsub.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        _, mask_thresh = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
        # foreground = cv2.bitwise_and(frame, frame, mask=mask_thresh)

        if cv2.countNonZero(mask_thresh) > 5000:
            idx.append(i)
    return idx

if __name__ == '__main__':
    path = "videos/stage31.mp4"
    print(detect_motion_events(path))

        