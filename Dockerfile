# Hugging Face Spaces (Docker SDK) / any container host: serves the Flask app with gunicorn on port 7860
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "cv_web_app:app"]
