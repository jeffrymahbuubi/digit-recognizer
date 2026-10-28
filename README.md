---
title: Digit Recognizer
sdk: docker
app_port: 7860
---

# Digit Recognizer web app

The course's Week 11 computer vision web app: draw a digit on a canvas, and an ONNX model run through OpenCV's DNN module recognises it.

| File | Purpose |
|---|---|
| `cv_web_app.py`, `templates/index.html` | Flask app (canvas page + `/upload` JSON endpoint) |
| `streamlit_app.py` | Streamlit version (upload an image of a digit) for Streamlit Community Cloud |
| `digit_model.py`, `model.onnx` | the model and its inference code, shared by both apps |
| `requirements.txt` | Python packages (headless OpenCV for servers) |
| `Procfile` | Heroku: `web: gunicorn cv_web_app:app` |
| `Dockerfile` | Hugging Face Spaces (Docker) or any container host: gunicorn on port 7860 |
| `samples/` | test images |

## Run locally
```
pip install -r requirements.txt
python cv_web_app.py              # Flask: http://127.0.0.1:5000
streamlit run streamlit_app.py    # Streamlit: http://localhost:8501
```

The block at the top of this file is the configuration for Hugging Face Spaces (Docker SDK, port 7860).
