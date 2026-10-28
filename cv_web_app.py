"""Digit Recognizer web app (Flask), the course's Week 11 app prepared for deployment.
Local:  python cv_web_app.py            -> http://127.0.0.1:5000
Server: gunicorn cv_web_app:app         (Procfile for Heroku; the Dockerfile runs it on port 7860 for Hugging Face Spaces)"""
import base64
import json
import os
import sys

import cv2
import numpy as np
from flask import Flask, render_template, request

import digit_model

# Define App
app = Flask(__name__, template_folder="templates")


# The home page is routed to index.html inside templates/
@app.route('/')
def index():
    return render_template('index.html')


# Handles the uploaded canvas image
@app.route('/upload', methods=["POST"])
def upload():
    # The data field holds a data URL: "data:image/jpeg;base64,<data>"
    data = request.form.get('data', '')
    if ',' not in data:
        return json.dumps({'success': False, 'error': 'no image data'}), 400, {'ContentType': 'application/json'}
    # The first part of the string only says what kind of file it is, so keep the data part
    data = data.split(',', 1)[1]
    # base64 decode, convert to a numpy array, decode the image in memory (nothing is written to disk)
    nparr = np.frombuffer(base64.decodebytes(data.encode()), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return json.dumps({'success': False, 'error': 'not an image'}), 400, {'ContentType': 'application/json'}

    class_id, confidence, _ = digit_model.predict(img)

    # Print results on the server side
    print("classId: {} confidence: {}".format(class_id, confidence), file=sys.stdout)
    # Return result as a json object
    return json.dumps({'success': True, 'class': class_id, 'confidence': confidence}), 200, {'ContentType': 'application/json'}


if __name__ == '__main__':
    # PORT is set by the hosting platform; locally the Flask default 5000 is used
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
