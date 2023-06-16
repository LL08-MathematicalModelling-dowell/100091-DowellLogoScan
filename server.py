from feature_extractor import FeatureExtractor
import urllib.request
from werkzeug.utils import secure_filename
from flask import Flask, request, flash, redirect, url_for, render_template, send_file , Response
import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
from pymongo import MongoClient
from gridfs import GridFS
from io import BytesIO
from bson import ObjectId
from moviepy.editor import VideoFileClip
import tempfile


UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'mp4'}
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

######################### Initialize connection with MongoDB ###########################
client = MongoClient("mongodb://localhost:27017/")

# Create database and collections
database = client['logoscan']
image_collection = database['images']
feature_collection = database['features']
fs = GridFS(database)

###################### Read images features from the DB #######################

# Read image features from MongoDB
fe = FeatureExtractor()
features = []
img_paths = []

# Iterate over the feature documents in the collection
for doc in feature_collection.find():
    # Retrieve the image_id and feature
    image_id = doc['image_id']
    feature = doc['feature']

    # Extract the image path
    img_path = "http://192.168.1.4:8000/image/" + str(image_id)

    # Append the feature and image path to the respective lists
    features.append(feature)
    img_paths.append(img_path)

# Convert the lists to numpy arrays
features = np.array(features)

#################### upload image to database #####################


# Initialize FeatureExtractor
fe1 = FeatureExtractor()

# Define Flask endpoint for uploading images and extracting features


@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Get uploaded image file, filename and extension
        image_file = request.files['image']
        filename = image_file.filename
        extension = filename.rsplit('.', 1)[1].upper()
        # Open the image using PIL
        image = Image.open(image_file)

        # Convert image to BytesIO object
        image_io = BytesIO()
        image.save(image_io, format=extension)
        image_io.seek(0)

        # Store image in MongoDB using GridFS
        image_id = fs.put(image_io, filename=image_file.filename)

        # Extract feature
        feature = fe1.extract(img=image)

        # Store feature in MongoDB
        feature_collection.insert_one({'image_id': image_id, 'feature': feature.tolist()})

        return 'Image uploaded and features extracted successfully!'

    return render_template('upload.html')


######################### get image with url #########################

# Define Flask route to serve the image
@app.route('/image/<image_id>')
def get_image(image_id):
    # Retrieve the image data from MongoDB
    image_data = fs.get(ObjectId(image_id)).read()
    file_name = database.fs.files.find_one({"_id": ObjectId(image_id)})["filename"]
    # get png or jpg part
    file_extension = file_name.split(".")[-1]

    # Return the image data as a response
    return Response(image_data, mimetype=f'image/{file_extension}')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


######################### uploading videos #########################


# Define Flask route for uploading videos and processing frames

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    filename = file.filename

    if not filename:
        flash('No video selected for uploading')
        return redirect(request.url)
    # Get file extension
    extension = filename.rsplit('.', 1)[1].upper()

    if file and allowed_file(file.filename):
        # Save the video file to MongoDB using GridFS
        video_id = fs.put(file, filename=file.filename, extension=extension)
        flash('Video successfully uploaded and processed')

        # Retrieve the video file from MongoDB using GridFS
        video = fs.get(video_id)
        frame_count = 0
        final_scores = []

        while True:
            try:
                video.seek(frame_count)
                img = Image.open(video)
                img = img.convert('RGB')
                query = fe.extract(img)
                dists = np.linalg.norm(features - query, axis=1)  # L2 distances to features
                ids = np.argsort(dists)[:30]  # Top 30 results
                scores = [str(img_paths[id]) for id in ids]
                final_scores.extend(scores)
                frame_count += 1

                # Save the frame image to MongoDB using GridFS
                frame_id = fs.put(img, filename=f"{video_id}_{frame_count}.jpg")
            except EOFError:
                break

        video.close()
        final_scores = list(set(final_scores))

        return {'filename': file.filename, 'scores': final_scores}

    else:
        flash('Allowed video types are mp4')
        return redirect(request.url)
    
######################### uploading videos #########################


######################### uploading videos #########################

# Define Flask route for uploading videos and processing frames
# @app.route('/upload', methods=['POST'])
# def upload_video():
#     if 'file' not in request.files:
#         flash('No file part')
#         return redirect(request.url)

#     file = request.files['file']

#     if file.filename == '':
#         flash('No video selected for uploading')
#         return redirect(request.url)

#     if file and allowed_file(file.filename):
#         # Save the video file to MongoDB using GridFS
#         video_id = fs.put(file, filename=file.filename)
#         flash('Video successfully uploaded and processed')

#         # Retrieve the video file from MongoDB using GridFS
#         video_file = fs.get(video_id)
        
#         # Create an in-memory file-like object from the video data
#         video_data = BytesIO(video_file.read())
        
#         cam = cv2.VideoCapture(video_data)# i want to change this part here to take the frames without using open cv
#         current_frame = 0
#         final_scores = []

#         while True:
#             ret, frame = cam.read()
#             if ret:
#                 img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#                 query = fe.extract(img)
#                 # L2 distances to features
#                 dists = np.linalg.norm(features - query, axis=1)
#                 ids = np.argsort(dists)[:30]  # Top 30 results
#                 scores = [str(img_paths[id]) for id in ids]
#                 final_scores.extend(scores)
#                 current_frame += 1

#                 # Save the frame image to MongoDB using GridFS
#                 frame_id = fs.put(frame, filename=f"{video_id}_{current_frame}.jpg")

#             else:
#                 break

#         cam.release()
#         cv2.destroyAllWindows()
#         final_scores = list(set(final_scores))

#         return {'filename': file.filename, 'scores': final_scores}

#     else:
#         flash('Allowed video types are mp4')
#         return redirect(request.url)

#########################################################

# Define Flask route for uploading videos and processing frames

# @app.route('/upload', methods=['POST'])
# def upload_video():
#     if 'file' not in request.files:
#         flash('No file part')
#         return redirect(request.url)

#     file = request.files['file']

#     if file.filename == '':
#         flash('No video selected for uploading')
#         return redirect(request.url)

#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)
#         flash('Video successfully uploaded and displayed below')

#         cam = cv2.VideoCapture(file_path)
#         current_frame = 0
#         final_scores = []

#         while True:
#             ret, frame = cam.read()
#             if ret:
#                 name = './data/frame' + str(current_frame) + '.jpg'
#                 cv2.imwrite(name, frame)
#                 img = Image.open(name)
#                 query = fe.extract(img)
#                 dists = np.linalg.norm(features - query, axis=1)  # L2 distances to features
#                 ids = np.argsort(dists)[:30]  # Top 30 results
#                 scores = [str(img_paths[id]) for id in ids]
#                 final_scores.extend(scores)
#                 current_frame += 1
#             else:
#                 break

#         cam.release()
#         cv2.destroyAllWindows()
#         final_scores = list(set(final_scores))

#         return {'filename': filename, 'scores': final_scores}

#     else:
#         flash('Allowed video types are mp4')
#         return redirect(request.url)

###########################################################


@app.route('/display/<filename>')
def display_video(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
