from ctypes import sizeof
import cv2
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse


from .image_comparison.main import ColorDescriptor, Searcher

def index(request):
    return JsonResponse({"message": "This is the backend api."})


from rest_framework.views import APIView
from rest_framework.renderers import TemplateHTMLRenderer
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from .serializers import LogoSerializer
import os
from dowell.settings import BASE_DIR, STATIC_ROOT

class LogoUploadView(APIView):
  parser_classes = (MultiPartParser, FormParser)

  #renderer_classes= [TemplateHTMLRenderer]
  #template_name = 'upload_logo.html'

  #def get(self, request, *args, **kwargs):
    #serializer = LogoSerializer(data=request.data)
    #return Response({'serializer': serializer})

  def post(self, request, *args, **kwargs):
    logo_serializer = LogoSerializer(data=request.data)

    if logo_serializer.is_valid():
      #print(request.FILES['image'])

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

      limit = 10
      reults = results[:limit]

      return JsonResponse({'results': results}, status=status.HTTP_201_CREATED)


    else:
      return JsonResponse({'error': logo_serializer.errors}, status=status.HTTP_400_BAD_REQUEST)