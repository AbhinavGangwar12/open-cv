import cv2 
import numpy as np

# load the model
try:
    net = cv2.dnn.readNetFromONNX("yolov8n.onnx")
except cv2.error:
    pass 

cap = cv2.VideoCapture(0)
ret, _ = cap.read()
if not ret:
    raise RuntimeError("Camera not found!")
cv2.namedWindow("YoloV8n", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YoloV8n", 800, 600)
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Camera not found!")
        break 

    org_h, org_w = frame.shape[:2]
    # create blob
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), (0, 0, 0), swapRB=True, crop=False)

    net.setInput(blob)
    outputs = net.forward()

    outputs = np.array(outputs)
    if len(outputs.shape) == 3:
        outputs = outputs[0]
    outputs = outputs.T
    boxes = []
    confidences = []
    class_ids = []

    x_scale = org_w / INPUT_WIDTH
    y_scale = org_h / INPUT_HEIGHT

    for detection in outputs:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            cx, cy, w, h = detection[0], detection[1], detection[2], detection[3]

            box_w = int(w * x_scale)
            box_h = int(h * y_scale)
            box_x = int((cx - w / 2) * x_scale)
            box_y = int((cy - h / 2) * y_scale)

            boxes.append([box_x, box_y, box_w, box_h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
    
    if len(indices) > 0:
        # Handle different OpenCV versions for NMS indices format
        indices = indices.flatten() if hasattr(indices, 'flatten') else indices
        for i in indices:
            idx = i[0] if isinstance(i, (list, np.ndarray)) else i 
            x, y, w, h = boxes[idx]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("YoloV8n", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


