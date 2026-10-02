import torch
import cv2
import numpy as np

def preprocess_for_pytorch(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (244, 244))
    rgb = cv2.normalize(rgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    rgb = np.transpose(rgb, (2, 0, 1))
    return torch.tensor(rgb)


print(preprocess_for_pytorch("images/sample.jpg").shape)