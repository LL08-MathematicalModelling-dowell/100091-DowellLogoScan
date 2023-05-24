import sys

from django.shortcuts import render, HttpResponseRedirect
from django.urls import reverse
from .forms import LogoUploadForm
from logoscan.settings import BASE_DIR
import os
from backend.image_comparison.main import ColorDescriptor, Searcher
from backend.serializers import LogoSerializer
from rest_framework import status
import cv2
import time
import imutils
####
import numpy as np
import csv

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
          results = searcher.search(features)


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


# def logo_upload_view2(request):
#   if request.method == 'POST':

#       start_time = time.time()
#       form = LogoUploadForm(request.POST, request.FILES)


#       if form.is_valid():

#         image_file = request.FILES['image']
#         logo_serializer = LogoSerializer(data={'image': image_file})

#         if logo_serializer.is_valid():
#             cd = ColorDescriptor((8, 12, 3))

#             # save the image as model in db
#             logo_serializer.save()
#             cd = ColorDescriptor((8, 12, 3))

#             #get query img path
#             img_name = logo_serializer.data['image'].replace('/media/', '')
#             media_path = os.path.join(BASE_DIR, 'media')
#             img_path = os.path.join(media_path, img_name)
#             #print(img_path)

#             # get the features of query image
#             query = cv2.imread(img_path)
#             features_query = cd.describe2(query)


#             # take all the files in ... dir and put their features in csv file for later use
#             directory = os.path.join(BASE_DIR, "media/logos")
#             output1 = open("index2.csv", "w")
#             for filename in os.scandir(directory):
#               filepath = os.path.join(directory, filename.name)
#               if filename.is_file():
#                 fileId= filename.name
#                 filesize = os.path.getsize(filename)
#                 if filesize!=0:
#                   image = cv2.imread(filepath)
#                   (h, w) = image.shape[:2]
#                   # image = imutils.resize(image, h, w)
#                   features = cd.describe2(image)
#                   features = [str(f) for f in features]
#                   output1.write("%s,%s\n" % (filename.name,",".join(features)))
#             output1.close()



#             # perform search
#             searcher = Searcher("index2.csv")
#             results = searcher.search(features_query)

#             context = {
#                 'compared_logo': img_name,
#                 'content': results,
#                 'form': form,
#                 'time_taken': time.time() - start_time
#               }
#             return render(request, 'upload-logo.html', context)
#   else:
#       form = LogoUploadForm()
#   return render(request,  'upload-logo.html', {'form': form})




