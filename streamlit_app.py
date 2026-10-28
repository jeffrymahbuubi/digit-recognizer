"""Digit Recognizer (Streamlit version of the course's Flask app, same ONNX model) for Streamlit Community Cloud.
Local: streamlit run streamlit_app.py   -> http://localhost:8501"""
import cv2
import numpy as np
import streamlit as st

import digit_model

st.title("Digit Recognizer")
st.write("Upload an image of one handwritten digit (0-9). Dark digits on a light background are inverted automatically, "
         "because the model expects a white digit on black, like the drawing canvas of the Flask app.")

img_file_buffer = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg", "bmp"])

if img_file_buffer is not None:
    # Read the file into a byte array and decode it with OpenCV (grayscale)
    raw_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    img = cv2.imdecode(raw_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        st.error("Could not read this file as an image.")
    else:
        invert = st.checkbox("Invert colours (dark digit on light paper)", value=bool(img.mean() > 127))
        model_input = 255 - img if invert else img
        digit, confidence, probs = digit_model.predict(model_input)

        col1, col2 = st.columns(2)
        col1.image(img, caption="Uploaded image", width="stretch")
        col2.image(cv2.resize(model_input, (140, 140), interpolation=cv2.INTER_AREA), caption="Model input (white on black)")
        st.subheader("Recognized digit: {}".format(digit))
        st.write("Confidence: {:.3f}".format(confidence))
        st.bar_chart({"probability": {str(i): float(p) for i, p in enumerate(probs)}})
else:
    st.info("Sample images are in the samples/ folder of the repository.")
