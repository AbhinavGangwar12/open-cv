import cv2 
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    h, w, c = frame.shape
    mid_h, mid_w = h // 2, w // 2



    q1 = frame[:mid_h, :mid_w]
    q2 = frame[:mid_h, mid_w:]
    q3 = frame[mid_h:, : mid_w]
    # q4 = frame[mid_h :,mid_w :]


    gray_q1 = cv2.cvtColor(q1, cv2.COLOR_BGR2GRAY)
    frame[:mid_h, :mid_w] = cv2.cvtColor(gray_q1, cv2.COLOR_GRAY2BGR)


    frame[:mid_h, mid_w:] = cv2.cvtColor(q2, cv2.COLOR_BGR2HSV)    
    frame[mid_h:, : mid_w] = cv2.cvtColor(q3, cv2.COLOR_BGR2RGB)

    cv2.line(frame, pt1=(mid_w, 0), pt2=(mid_w, h), color=(255, 255, 255), thickness=2)
    cv2.line(frame, pt1=(0, mid_h), pt2=(w, mid_h), color=(255, 255, 255), thickness=2)
    cv2.imshow("Activity", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting....")
        break

cap.release()
cv2.destroyAllWindows()
