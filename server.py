import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
# from PIL import Image
# from feature_extractor import FeatureExtractor
# from pathlib import Path
# import numpy as np


import os
import cv2
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
    

app = Flask(__name__)


UPLOAD_FOLDER = 'static/uploads/'

# app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
features = np.array(features)


@app.route('/photo', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:30]  # Top 30 results
        scores = [(dists[id], img_paths[id]) for id in ids]

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')




	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_video():
	try:
		
		if not os.path.exists('data'):
			os.makedirs('data')
	except Exception as e:
		print ('Error: Creating directory of data')
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	else:
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_video filename: ' + filename)
		flash('Video successfully uploaded and displayed below')
		cam = cv2.VideoCapture(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		currentframe=0
		final_score=[]
		final_score1 = []
		final_score2=[]
		while(True):
				ret,frame = cam.read()
				if ret:
					name = './data/frame' + str(currentframe) + '.jpg'
				
					cv2.imwrite(name, frame)
					img = Image.open(name) 
					query = fe.extract(img)
					
					dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
					ids = np.argsort(dists)[:30]  # Top 30 results
					scores = [ img_paths[id] for id in ids]
	                # scores = [(dists[id], img_paths[id]) for id in ids]
					# final_score1.extend([scores[0]])
					# final_score2.extend([scores[1]])
					final_score.extend([scores[0],scores[1]])
					
					# final_score = final_score + list(set(scores)-list(final_score))
			
					currentframe += 1
				else:
					break
		cam.release()
		cv2.destroyAllWindows()
		# final_score1 = list(set(final_score1))
		# final_score2 = list(set(final_score2))
		print("________________")
		# print(final_score1)
		# print(final_score2)
		# final_score = list(set(final_score1 + final_score2))
		final_score = list(set(final_score))
		print(final_score)
		
		return render_template('index.html', filename=filename, scores=final_score)
	# return render_template('upload.html', filename=filename)

@app.route('/display/<filename>')
def display_video(filename):
	#print('display_video filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__=="__main__":
    # fe = FeatureExtractor()

    # for img_path in sorted(Path("./static/img").glob("*.jpg")):
    #     print(img_path)  # e.g., ./static/img/xxx.jpg
    #     feature = fe.extract(img=Image.open(img_path))
    #     feature_path = Path("./static/feature") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
    #     np.save(feature_path, feature)
    # app.run("0.0.0.0")
    	app.run(debug=True)

