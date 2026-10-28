import cv2 

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Video not found!")
prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("Coding Challenge", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Coding Challenge", 500, 500)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Video not found!")

    curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(curr, prev)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    cv2.imshow("Coding Challenge", frame)
    prev = curr 
    if cv2.countNonZero(thresh) > 10000:
        cv2.imwrite("motion_detected.jpg", frame)
        break 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()