from feature_extractor import FeatureExtractor
import urllib.request
from werkzeug.utils import secure_filename
from flask import Flask, request, flash, redirect, url_for
import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image


UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'mp4'}
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append("http://192.168.1.6:8000/"+str(Path("./static/img") / (feature_path.stem + ".jpg")))
features = np.array(features)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No video selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        flash('Video successfully uploaded and displayed below')

        cam = cv2.VideoCapture(file_path)
        current_frame = 0
        final_scores = []

        while True:
            ret, frame = cam.read()
            if ret:
                name = './data/frame' + str(current_frame) + '.jpg'
                cv2.imwrite(name, frame)
                img = Image.open(name)
                query = fe.extract(img)
                dists = np.linalg.norm(features - query, axis=1)  # L2 distances to features
                ids = np.argsort(dists)[:30]  # Top 30 results
                scores = [str(img_paths[id]) for id in ids]
                final_scores.extend(scores)
                current_frame += 1
            else:
                break

        cam.release()
        cv2.destroyAllWindows()
        final_scores = list(set(final_scores))

        return {'filename': filename, 'scores': final_scores}

    else:
        flash('Allowed video types are mp4')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_video(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
