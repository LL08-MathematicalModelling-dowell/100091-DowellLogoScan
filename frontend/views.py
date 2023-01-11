import sys

from django.shortcuts import render, HttpResponseRedirect
from django.urls import reverse
from .forms import LogoUploadForm
from logoscan.settings import BASE_DIR
import os
#from backend.image_comparison.main import ColorDescriptor, Searcher
from backend.serializers import LogoSerializer
from rest_framework import status
import cv2
import time
import imutils

####
import cv2
import numpy as np
import imutils
import csv
from heapq import nlargest

####

class ColorDescriptor:
	def __init__(self, bins):
		# store the number of bins for the 3D histogram
		self.bins = bins
	def describe(self, image):
		# convert the image to the HSV color space and initialize
		# the features used to quantify the image
		#print("this is describe function")
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		features = []
		# grab the dimensions and compute the center of the image
		(h, w) = image.shape[:2]
		(cX, cY) = (int(w * 0.5), int(h * 0.5))
				# divide the image into four rectangles/segments (top-left,
		# top-right, bottom-right, bottom-left)
		segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),(0, cX, cY, h)]
		# construct an elliptical mask representing the center of the
		# image
		(axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
		ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
		cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
		# loop over the segments
		for (startX, endX, startY, endY) in segments:
			# construct a mask for each corner of the image, subtracting
			# the elliptical center from it
			cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
			cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
			cornerMask = cv2.subtract(cornerMask, ellipMask)
			# extract a color histogram from the image, then update the
			# feature vector
			hist = self.histogram(image, cornerMask)
			features.extend(hist)
		# extract a color histogram from the elliptical region and
		# update the feature vector
		hist = self.histogram(image, ellipMask)
		features.extend(hist)
		# return the feature vector
		return features

	def describe2(self, image):
		# convert the image to the HSV color space and initialize
		# the features used to quantify the image
		#print("this is describe function")
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		features = []
		# grab the dimensions and compute the center of the image
		(h, w) = image.shape[:2]
		(cX, cY) = (int(w * 0.5), int(h * 0.5))
				# divide the image into four rectangles/segments (top-left,
		# top-right, bottom-right, bottom-left)
# 		segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),(0, cX, cY, h)]
		# construct an elliptical mask representing the center of the
		# image
		(axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
		ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
		cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
		# loop over the segments
# 		for (startX, endX, startY, endY) in segments:
# 			# construct a mask for each corner of the image, subtracting
# 			# the elliptical center from it
# 			cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
# 			cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
# 			cornerMask = cv2.subtract(cornerMask, ellipMask)
# 			# extract a color histogram from the image, then update the
# 			# feature vector
# 			hist = self.histogram(image, cornerMask)
# 			features.extend(hist)
		# extract a color histogram from the elliptical region and
		# update the feature vector
		hist = self.histogram(image, ellipMask)
		features.extend(hist)
		# return the feature vector
		return features

	def histogram(self, image, mask):
		# extract a 3D color histogram from the masked region of the
		# image, using the supplied number of bins per channel
 		#print("this is  histogrm class")
		hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
			[0, 180, 0, 256, 0, 256])
		# normalize the histogram if we are using OpenCV 2.4
		if imutils.is_cv2():
			hist = cv2.normalize(hist).flatten()
		# otherwise handle for OpenCV 3+
		else:
			hist = cv2.normalize(hist, hist).flatten()
		# return the histogram
		return hist


# searcher algorithm


class Searcher:
	def __init__(self, indexPath):
		# store our index path
		self.indexPath = indexPath
	def search(self, queryFeatures, limit = 3):
		# initialize our dictionary of results
		results = {}
		#print("this is searcher function")
				# open the index file for reading
		with open(self.indexPath) as f:
			# initialize the CSV reader
			reader = csv.reader(f)
			# loop over the rows in the index
			for row in reader:
				# parse out the image ID and features, then compute the
				# chi-squared distance between the features in our index
				# and our query features
				features = [float(x) for x in row[1:]]
				d = self.chi2_distance(features, queryFeatures)
				# now that we have the distance between the two feature
				# vectors, we can udpate the results dictionary -- the
				# key is the current image ID in the index and the
				# value is the distance we just computed, representing
				# how 'similar' the image in the index is to our query
				results[row[0]] = d
			# close the reader
			f.close()
		# sort our results, so that the smaller distances (i.e. the
		# more relevant images are at the front of the list)
		results = sorted([(v, k) for (k, v) in results.items()])
		# return our (limited) results
		return results[0:2]
		#return results



	def chi2_distance(self, histA, histB, eps = 1e-10):
		# compute the chi-squared distance
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])
		# return the chi-squared distance
		return d

####

def logo_upload_view(request):
  
  if request.method == 'POST':
    start_time = time.time()
    form = LogoUploadForm(request.POST, request.FILES)


    if form.is_valid():
      image_file = request.FILES['image']
      logo_serializer = LogoSerializer(data={'image': image_file})


      if logo_serializer.is_valid():
          cd = ColorDescriptor((8, 12, 3))

          # take all the files in ... dir and put their features in csv file for later use
          directory = os.path.join(BASE_DIR, "media/logos")
          output1 = open("index1.csv", "w")
          for filename in os.scandir(directory):
            filepath = os.path.join(directory, filename.name)
            if filename.is_file():
              fileId= filename.name
              filesize = os.path.getsize(filename)
              if filesize!=0:
                image = cv2.imread(filepath)
                features = cd.describe(image)
                features = [str(f) for f in features]
                output1.write("%s,%s\n" % (filename.name, ",".join(features)))
          output1.close()

          # save the image as model in db
          logo_serializer.save()
          cd = ColorDescriptor((8, 12, 3))

          #get query img path
          img_name = logo_serializer.data['image'].replace('/media/', '')
          media_path = os.path.join(BASE_DIR, 'media')
          img_path = os.path.join(media_path, img_name)
          #print(img_path)

          # get the features of query image
          query = cv2.imread(img_path)
          features = cd.describe(query)
          # perform search
          searcher = Searcher("index1.csv")
          results = searcher.search(features, limit=1)

          context = {
              'compared_logo': img_name,
              'content': results,
              'form': form,
              'time_taken': time.time() - start_time
            }
			

          return render(request, 'upload-logo.html', context)
  else:
    form = LogoUploadForm()
  return render(request,  'upload-logo.html', {'form': form})


def logo_upload_view2(request):
  if request.method == 'POST':

      start_time = time.time()
      form = LogoUploadForm(request.POST, request.FILES)


      if form.is_valid():
      
        image_file = request.FILES['image']
        logo_serializer = LogoSerializer(data={'image': image_file})

        if logo_serializer.is_valid():
            cd = ColorDescriptor((8, 12, 3))

            # save the image as model in db
            logo_serializer.save()
            cd = ColorDescriptor((8, 12, 3))

            #get query img path
            img_name = logo_serializer.data['image'].replace('/media/', '')
            media_path = os.path.join(BASE_DIR, 'media')
            img_path = os.path.join(media_path, img_name)
            #print(img_path)

            # get the features of query image
            query = cv2.imread(img_path)
            features_query = cd.describe2(query)


            # take all the files in ... dir and put their features in csv file for later use
            directory = os.path.join(BASE_DIR, "media/logos")
            output1 = open("index2.csv", "w")
            for filename in os.scandir(directory):
              filepath = os.path.join(directory, filename.name)
              if filename.is_file():
                fileId= filename.name
                filesize = os.path.getsize(filename)
                if filesize!=0:
                  image = cv2.imread(filepath)
                  (h, w) = image.shape[:2]
                  # image = imutils.resize(image, h, w)
                  features = cd.describe2(image)
                  features = [str(f) for f in features]
                  output1.write("%s,%s\n" % (filename.name,",".join(features)))
            output1.close()



            # perform search
            searcher = Searcher("index2.csv")
            results = searcher.search(features_query, limit=3)

            context = {
                'compared_logo': img_name,
                'content': results,
                'form': form,
                'time_taken': time.time() - start_time
              }
            return render(request, 'upload-logo.html', context)
  else:
      form = LogoUploadForm()
  return render(request,  'upload-logo.html', {'form': form})
