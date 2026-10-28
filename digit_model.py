"""Digit recognition with the course's ONNX model through OpenCV's DNN module (shared by the Flask and Streamlit apps)."""
import os

import cv2
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.onnx")

# Load Digit Recognition model (once, at import)
net = cv2.dnn.readNetFromONNX(MODEL_PATH)


def softmax(x):
    """Compute softmax values for a vector of scores."""
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def predict(img_gray):
    """img_gray: 2-D uint8 image, white digit on a black background (like the drawing canvas).
    Returns (digit, confidence, probabilities of the 10 classes)."""
    if img_gray is None or img_gray.ndim != 2 or img_gray.size == 0:
        raise ValueError("expected a non-empty grayscale image")
    # Create a 4D blob from image: scale to [0, 1], resize to the 28x28 input of the model
    blob = cv2.dnn.blobFromImage(img_gray, 1 / 255, (28, 28))
    net.setInput(blob)
    probs = softmax(net.forward().flatten())
    class_id = int(np.argmax(probs))
    return class_id, float(probs[class_id]), probs
