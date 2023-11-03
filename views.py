# videoupload/views.py
import cv2
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from .models import Video,Video1
from .serializers import VideoSerializer,VideoSerializer1,VideoSerializer2
from django.http import  JsonResponse
import os
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from statistics import mean
present_dir =os.path.dirname(os.path.abspath(__file__))
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from pathlib import Path
from PIL import Image
import pymongo
import gridfs
import numpy as np
import math



#################### new project ####################

import os

from django.http import HttpResponse

from rest_framework.parsers import MultiPartParser, FormParser


from bson.objectid import ObjectId

from io import BytesIO


######## new project start ##############
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
######## new project end ##############


# client = pymongo.MongoClient("mongodb+srv://ahmed:9izIB3wFIls8vins@cluster0.j9hy3nb.mongodb.net/?retryWrites=true&w=majority")
client = pymongo.MongoClient("mongodb+srv://new_project:new_project@cluster0.yhltv9t.mongodb.net/?retryWrites=true&w=majority")


# Create database and collections
database = client['logoscan']

image_features_collection = database.image_features
user_image_features_collection = database.admin_frame
# Create compound index on category, product, and brand fields
image_features_collection.create_index([('category', pymongo.ASCENDING),
                                  ('product', pymongo.ASCENDING), ('brand', pymongo.ASCENDING)])

user_image_features_collection.create_index([('category', pymongo.ASCENDING),
                                  ('product', pymongo.ASCENDING), ('brand', pymongo.ASCENDING)])


import os
# from feature_extractor import FeatureExtractor
present_dir =os.path.dirname(os.path.abspath(__file__))


class FeatureExtractor:
    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    def extract(self, img):
        """
        Extract a deep feature from an input image
        Args:
            img: from PIL.Image.open(path) or tensorflow.keras.preprocessing.image.load_img(path)

        Returns:
            feature (np.ndarray): deep feature with the shape=(4096, )
        """
        print("extracting feature of image")
        img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel

        feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
        print("extracting feature of image is done ")
        return feature / np.linalg.norm(feature)  # Normalize

    def extract1(self, img_path):
        """
        Extract a deep feature from an input image
        Args:
            img: from PIL.Image.open(path) or tensorflow.keras.preprocessing.image.load_img(path)

        Returns:
            feature (np.ndarray): deep feature with the shape=(4096, )
        """
        print("extract1 function : extracting feature of image")
        img =Image.open(img_path)
        img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel
        feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
        print("extract1 extracting feature of image is done ")
        return feature / np.linalg.norm(feature)  # Normalize



def index(request):
    return JsonResponse({"message": "This is the backend api."})



@method_decorator(csrf_exempt, name='dispatch')
class VideoUploadView1(View):
    def post(self, request, *args, **kwargs):
        features = []
        img_paths = []
        # feature = f'{present_dir}/feature'
        img = f'{present_dir}/img'


        count = 0
        # for itr in image_features_collection.find():
        #     count = count+1
        #     img_paths.append(itr['id'])
        #     features.append(itr['feature'])
        # features = np.array(features)
        print(request)
        upload_path = f'{present_dir}/data'
        upload_image_path = f'{present_dir}/img_frame'

        try:
            if not os.path.exists(upload_path):
                os.makedirs(upload_path)
        except Exception as e:
            print('Error: Creating directory of data')

        try:
            if not os.path.exists(upload_image_path):
                os.makedirs(upload_image_path)
        except Exception as e:
            print('Error: Creating directory of img_frame')


        if 'video' in request.FILES:
            video_file = request.FILES['video']

            # Adjust this path to where you want to store the videos

            video_path = os.path.join(upload_path, "upload.mp4")

            with open(video_path, 'wb') as f:
                for chunk in video_file.chunks():
                    f.write(chunk)
            cam = cv2.VideoCapture(video_path)
            return JsonResponse({'message': 'Video uploaded successfully'})
        else:
            return JsonResponse({'error': 'No video file provided'}, status=400)



# admin upload view
@method_decorator(csrf_exempt, name='dispatch')
class VideoUploadView(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request):
        try :
            file_serializer = VideoSerializer1(data=request.data)

            if file_serializer.is_valid():
                file_serializer.save()

                return Response('file_serializer.data', status=status.HTTP_201_CREATED)
            else:
                return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=200)




# admin upload view
@method_decorator(csrf_exempt, name='dispatch')
class VideoUploadViewFastUpload(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request):
        try :
            file_serializer = VideoSerializer2(data=request.data)

            if file_serializer.is_valid():
                file_serializer.save()

                return Response('file_serializer.data', status=status.HTTP_201_CREATED)
            else:
                return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=200)




# image_features_collection = database.image_features
# fs = gridfs.GridFS(database)

@method_decorator(csrf_exempt, name='dispatch')
class VideoUploadView_progress(View):
    def post(self, request, *args, **kwargs):
        features = []
        img_paths = []
        # feature = f'{present_dir}/feature'
        img = f'{present_dir}/img'


        count = 0
        upload_path = f'{present_dir}/data'
        upload_image_path = f'{present_dir}/img_frame'

        try:
            if not os.path.exists(upload_path):
                os.makedirs(upload_path)
        except Exception as e:
            print('Error: Creating directory of data')

        try:
            if not os.path.exists(upload_image_path):
                os.makedirs(upload_image_path)
        except Exception as e:
            print('Error: Creating directory of img_frame')


        if 'video' in request.FILES:
            video_file = request.FILES['video']

            # Adjust this path to where you want to store the videos

            video_path = os.path.join(upload_path, "upload.mp4")

            with open(video_path, 'wb') as f:
                for chunk in video_file.chunks():
                    f.write(chunk)
            cam = cv2.VideoCapture(video_path)
            currentframe=0
            count_frame_buffer =0
            while(True):

                ret,frame = cam.read()
                count_frame_buffer = count_frame_buffer  + 1
                if count_frame_buffer%2 ==0:
                    print(f"---------->{count_frame_buffer}<-----------")
                    if ret:
                            # name = './data/frame' + str(currentframe) + '.jpg'
                            name = f"{upload_image_path}/{currentframe}.jpg"
                            # upload_image_path =os.path.join(upload_image_path, name)
                            cv2.imwrite(name, frame)

                            currentframe += 1
                    else:
                            break
            cam.release()
            cv2.destroyAllWindows()
            return JsonResponse({'message': "video frames extracted..!"})



        else:
            return JsonResponse({'error': 'No video file provided'}, status=400)





# test auth endpoint
class AdminAuth(APIView):
    def post(self, request, *args, **kwargs):
        username = request.POST.get('username')
        password = request.POST.get('password')
        print(username, password)
        if username == password == 'dowell':
            return JsonResponse({'authenticated': True})
        else:
            return JsonResponse({'authenticated': False})




present_dir =os.path.dirname(os.path.abspath(__file__))


fs = gridfs.GridFS(database)

fe = FeatureExtractor()


def calculate_chi2_probability1(chi2_value, degrees_of_freedom):
	    # Calculate the probability using chi-squared value and degrees of freedom
	    from scipy.stats import chi2
	    confidence_level = 0.95  # 95% confidence interval
	    critical_value = chi2.ppf(confidence_level, df=degrees_of_freedom)
	    probability = 1 - chi2.cdf(chi2_value, df=degrees_of_freedom)
	    return probability

@method_decorator(csrf_exempt, name='dispatch')
class VideoUploadViewFrames(View):
    def post(self, request, *args, **kwargs):
        features = []
        img_paths = []
        # feature = f'{present_dir}/feature'
        img = f'{present_dir}/img'


        count = 0
        for itr in image_features_collection.find():
            count = count+1
            try:
                itr_id = itr['id']
                itr_feature = itr['feature']
                img_paths.append(itr_id)
                features.append(itr_feature)
            except Exception as e:
                pass
        features = np.array(features)
        print(request)
        upload_path = f'{present_dir}/data'
        upload_image_path = f'{present_dir}/img_frame'

        try:
            if not os.path.exists(upload_path):
                os.makedirs(upload_path)
        except Exception as e:
            print('Error: Creating directory of data')

        try:
            if not os.path.exists(upload_image_path):
                os.makedirs(upload_image_path)
        except Exception as e:
            print('Error: Creating directory of img_frame')

        category = request.POST.get('category')
        product = request.POST.get('product')
        brand = request.POST.get('brand')

        if 'video' in request.FILES:
            video_file = request.FILES['video']

            # Adjust this path to where you want to store the videos

            video_path = os.path.join(upload_path, "upload.mp4")

            with open(video_path, 'wb') as f:
                for chunk in video_file.chunks():
                    f.write(chunk)
            cam = cv2.VideoCapture(video_path)
            currentframe=0
            final_score=[]
            final_score1 = []
            final_score2=[]
            count_frame_buffer =0
            results={}
            degrees_of_freedom = 8
            res_prob = []
            while(True):

                ret,frame = cam.read()
                count_frame_buffer = count_frame_buffer  + 1
                if count_frame_buffer%2 ==0:
                    print(f"---------->{count_frame_buffer}<-----------")
                    if ret:
                            # name = './data/frame' + str(currentframe) + '.jpg'
                            name = f"{upload_image_path}/currentframe.jpg"
                            # upload_image_path =os.path.join(upload_image_path, name)
                            cv2.imwrite(name, frame)
                            img = Image.open(name)
                            query = fe.extract(img)

                            find_one = image_features_collection.find_one({"feature":query.tolist()})

                            try:
                                dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
                                ids = np.argsort(dists)[:10]  # Top 20 results
                                scores = [ img_paths[id] for id in ids]

                                dists_score = [(dists[id], img_paths[id]) for id in ids]
                                print(f"results = {results}")

                                for id in ids:
                                    print(f"---------------------------->probdist{calculate_chi2_probability1(dists[id], degrees_of_freedom)}")
                                    if img_paths[id] in results :

                                        print(f"inside iff-{img_paths[id]} ---  {results[img_paths[id]]} value ends here {calculate_chi2_probability1(dists[id], degrees_of_freedom)}")
                                        results[img_paths[id]].append(float(calculate_chi2_probability1(dists[id], degrees_of_freedom)))
                                        print("------------inside if------------")
                                    else:
                                        print(f"----------inside else----------")
                                        results[img_paths[id]] = []+ [calculate_chi2_probability1(dists[id], degrees_of_freedom)]
                                        print(f"inside else----={img_paths[id]}  ----{calculate_chi2_probability1(dists[id], degrees_of_freedom)}  value ends here ")


                                final_score.extend([scores[0],scores[1]])
                                # final_score = final_score + list(set(scores)-list(final_score))
                            except Exception as e:
                                    print(e)

                            currentframe += 1
                    else:
                            break
            cam.release()
            cv2.destroyAllWindows()


            print("_________________________________________________________________________________")
            print(results)
            print("_________________________________________________________________________________")
            final_score = list(set(final_score))
            print(final_score)
            fs = gridfs.GridFS(database)
            images = []

            for keys in results:
                try:
                    results[keys]=mean(results[keys])
                except Exception as e:
                    results[keys]=0.0
            print(20*"{category}")
            if final_score :
                # return JsonResponse({'message': final_score})
                return JsonResponse({'message': results})
            else :
                return JsonResponse({'images': "no images found"})



        else:
            return JsonResponse({'error': 'No video file provided'}, status=400)



# this is the admin logo upload view.
@method_decorator(csrf_exempt, name='dispatch')
class LogoImageUploadView(View):
    def post(self, request, *args, **kwargs):
        print('working')
        try:

            if 'image' in request.FILES:
                image_file = request.FILES['image']
                category = request.POST.get('category')
                product = request.POST.get('product')
                brand = request.POST.get('brand')
                flag = request.POST.get('flag')

                # Adjust this path to where you want to store the images
                upload_image_path = f'{present_dir}/admin_img_frame'
                if not os.path.exists(upload_image_path):
                    os.makedirs(upload_image_path)

                # Save the image
                image_path = os.path.join(upload_image_path, "admin_upload.jpg")
                with open(image_path, 'wb') as f:
                    for chunk in image_file.chunks():
                        f.write(chunk)

                # Extract features
                img = Image.open(image_path)
                query = fe.extract(img)

                # Check if features already exist in the collection
                find_one = image_features_collection.find_one({"feature": query.tolist()})

                if not find_one:
                    # Store the image in GridFS
                    fs = gridfs.GridFS(database)
                    with open(image_path, 'rb') as f:
                        contents = f.read()
                        image_id = fs.put(contents, filename="admin_tempname",
                                         **{
                                             'category': category,
                                             'product': product,
                                             'brand': brand,
                                             'flag': flag
                                         }
                                         )

                    # Store features in MongoDB
                    data = {
                        'id': str(image_id),
                        'feature': query.tolist(),
                        'category': category,
                        'product': product,
                        'brand': brand,
                        'flag': flag
                    }
                    image_features_collection.insert_one(data)

                    return JsonResponse({'message': 'Image and features uploaded successfully'})
                else:
                    return JsonResponse({'message': 'Features already exist for this image'})

            else:
                return JsonResponse({'error': 'No image file provided'}, status=400)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)





class ImageAPIView(APIView):
    def get(self, request, id):
        # Connect to MongoDB
        fs = gridfs.GridFS(database)
        # Retrieve the image data from MongoDB
        try:
            image_data = fs.get(ObjectId(id))
            # fileName = database.fs.files.find_one(
            #     {
            #         "_id": ObjectId(id)
            #     }
            # )["name"]
            # get png or jpg part
            # if fileName:
            #     fileExtension = fileName.split(".")[-1]
            fileExtension = "test"
            return HttpResponse(image_data, content_type=f'image/{fileExtension}')

        except Exception as e:
            return JsonResponse({"Error Code 404":str(e)}, status=status.HTTP_404_NOT_FOUND)

        # Serialize the image data to return it as a response
        # return Response({'image_data': image_data})




class ReviewsView(APIView):
    def get(self, request, *args, **kwargs):
        # Extract the image id from the request GET parameters
        image_id = request.GET.get('image_id')
        if not image_id:
            return JsonResponse({'error': 'Missing image id'}, status=400)

        # finding the db collections
        fs_files = database['fs.files']
        reviews_collection = database['reviews']

        # Find the document with the given ID in the fs.files collection
        file_doc = fs_files.find_one({'_id': ObjectId(image_id)})
        if not file_doc:
            return JsonResponse({'error': 'File not found.'}, status=404)

        # Get the category, product, and brand from the file document
        category = file_doc.get('category')
        product = file_doc.get('product')
        brand = file_doc.get('brand')

        # Find the reviews with the same category, product, and brand in the reviews collection
        reviews = []
        for review in reviews_collection.find({'category': category, 'product': product, 'brand': brand}):
            reviews.append({'username': review.get('username'),
                           'feedback': review.get('feedback')})

        # Return the list of reviews
        return JsonResponse({'reviews': reviews, 'category': category, 'product': product, 'brand': brand})


class UserReview(APIView):

    def post(self, request, *args, **kwargs):
        # Extract the review data from the request POST parameters
        category = request.POST.get('category')
        product = request.POST.get('product')
        brand = request.POST.get('brand')
        username = request.POST.get('username')
        feedback = request.POST.get('feedback')

        # Check if all required parameters are present
        if not all([category, product, brand, username, feedback]):
            return JsonResponse({'error': 'Missing review data.'}, status=400)

        reviews_collection = database['reviews']

        # Insert the review into the reviews collection
        review_doc = {
            'category': category,
            'product': product,
            'brand': brand,
            'username': username,
            'feedback': feedback
        }
        result = reviews_collection.insert_one(review_doc)

        # Return a success response
        return JsonResponse({'message': 'Review submitted successfully.', 'review_id': str(result.inserted_id)})




class DropDownMenuData(APIView):
    def get(self, request, *args, **kwargs):

        # Get category, product, and brand values from request
        category = request.GET.get('category')
        product = request.GET.get('product')
        brand = request.GET.get('brand')

        # Define query object based on request parameters
        query = {}
        if category:
            query['category'] = category
        if product:
            query['product'] = product
        if brand:
            query['brand'] = brand

        # Query distinct categories, products, and brands based on query object
        categories = image_features_collection.distinct('category', query)
        products = image_features_collection.distinct('product', query)
        brands = image_features_collection.distinct('brand', query)

        # # Add "No Selection" to the beginning of each list
        # categories.insert(0, "No Selection")
        # products.insert(0, "No Selection")
        # brands.insert(0, "No Selection")

        # Return response as JSON
        data = {
            'categories': categories,
            'products': products,
            'brands': brands
        }
        return JsonResponse(data)



def calculate_chi2_probability(chi2_value, degrees_of_freedom):
	    # Calculate the probability using chi-squared value and degrees of freedom
	    from scipy.stats import chi2
	    confidence_level = 0.95  # 95% confidence interval
	    critical_value = chi2.ppf(confidence_level, df=degrees_of_freedom)
	    probability = 1 - chi2.cdf(chi2_value, df=degrees_of_freedom)
	    return probability

@method_decorator(csrf_exempt, name='dispatch')
def image_comparision(request):
    if request.method == 'POST':


        probability_cutoff = 0.5
        file = request.FILES['query_img']
        category = request.POST.get('category')
        product = request.POST.get('product')
        brand = request.POST.get('brand')

        # Save query image
        img = Image.open(file)  # PIL image
        print(file.name)
        filename = file.name.split('.')[0]
        print(20*"*1*",filename)

        # uploaded_img_path = os.path.join(present_dir)
        file_path_name = os.path.join(present_dir,"test")

        uploaded_img_path = file_path_name+".jpg"

        img.save(uploaded_img_path)
        print(20*"*2*",uploaded_img_path)



    #     # Run search

        features = []
        img_paths =[]
        count = 0
        for itr in image_features_collection.find():
            count = count+1
            try:
                img_paths.append(itr['id'])
                features.append(itr['feature'])
            except Exception as e:
                pass
        query = fe.extract(img=Image.open(uploaded_img_path))



        try:
            dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
            ids = np.argsort(dists)[:30]  # Top 30 results
            scores = [(dists[id], img_paths[id]) for id in ids]
            results={}
            degrees_of_freedom = 8
            res_prob = []
            for id in ids:
                results[img_paths[id]] = calculate_chi2_probability(dists[id], degrees_of_freedom)
                res_prob.append(calculate_chi2_probability(dists[id], degrees_of_freedom))


            print(100*"e",res_prob)
            print(100*"e",results)
            find_one = image_features_collection.find_one({"feature":query.tolist()})

            if probability_cutoff > max(res_prob):
            # if not find_one :
                fs = gridfs.GridFS(database)
                with open(uploaded_img_path,'rb') as f:
                    contents = f.read()
                video_id = fs.put(contents,filename ="tempname",
                                    **{
                                            'category': category,
                                            'product': product,
                                            'brand': brand,
                                            'flag': 'user_upload'
                                        }
                                        )
                data ={'id':str(video_id),
                                        'feature':query.tolist(),
                                         'category': category,
                                        'product': product,
                                        'brand': brand,
                                        'flag': 'user_upload'
                                        }
                image_features_collection.insert_one(data)

                return JsonResponse({'message': f"No image found , probability is less than{max(res_prob)} {probability_cutoff} ,{res_prob},image is saved sucessfully"})


            return JsonResponse({'message': results})
        except Exception as e:
                    return JsonResponse({'message': f'{e}'})




    #     features = []
    #     img_paths = []
    #     for feature_path in Path(f"{present_dir}/static/feature").glob("*.npy"):
    #         print("bssssssssssssside for loopppppppppppppp")
    #         features.append(np.load(feature_path))
    #         img_paths.append(Path(f"{present_dir}/static/img") / (feature_path.stem + ".jpg"))
    #     features = np.array(features)
    #     print(20*"*7*")
    #     print("---feature>",features)
    #     dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
    #     ids = np.argsort(dists)[:30]  # Top 30 results
    #     scores = [(dists[id], img_paths[id]) for id in ids]
    #     print(scores)
    #     # return HttpResponse(f"this is imgpath project {scores} ")
    #     scores =['t10.jpg']
    #     uploaded_img_path = 't10.jpg'

    #     return render(request,'index.html',
    #                           { "query_path":uploaded_img_path,
    #                           "scores":scores})
    # else:
    #     return render(request,'index.html')



@method_decorator(csrf_exempt, name='dispatch')
class VideoUploadViewFramesSpeed(View):
    def post(self, request, *args, **kwargs):
        features = []
        img_paths = []
        # feature = f'{present_dir}/feature'
        img = f'{present_dir}/img'


        count = 0
        for itr in image_features_collection.find():
            count = count+1
            try:
                itr_id = itr['id']
                itr_feature = itr['feature']
                img_paths.append(itr_id)
                features.append(itr_feature)
            except Exception as e:
                pass
        features = np.array(features)
        print(request)
        upload_path = f'{present_dir}/data'
        upload_image_path = f'{present_dir}/img_frame'

        try:
            if not os.path.exists(upload_path):
                os.makedirs(upload_path)
        except Exception as e:
            print('Error: Creating directory of data')

        try:
            if not os.path.exists(upload_image_path):
                os.makedirs(upload_image_path)
        except Exception as e:
            print('Error: Creating directory of img_frame')

        category = request.POST.get('category')
        product = request.POST.get('product')
        brand = request.POST.get('brand')

        if 'video' in request.FILES:
            video_file = request.FILES['video']

            # Adjust this path to where you want to store the videos

            video_path = os.path.join(upload_path, "upload.mp4")

            with open(video_path, 'wb') as f:
                for chunk in video_file.chunks():
                    f.write(chunk)
            cam = cv2.VideoCapture(video_path)
            currentframe=0
            final_score=[]
            final_score1 = []
            final_score2=[]
            count_frame_buffer =0
            results={}
            degrees_of_freedom = 8
            res_prob = []
            while(True):

                ret,frame = cam.read()
                count_frame_buffer = count_frame_buffer  + 1
                if count_frame_buffer%2 ==0:
                    print(f"---------->{count_frame_buffer}<-----------")
                    if ret:
                            # name = './data/frame' + str(currentframe) + '.jpg'
                            name = f"{upload_image_path}/currentframe.jpg"
                            # upload_image_path =os.path.join(upload_image_path, name)
                            cv2.imwrite(name, frame)
                            img = Image.open(name)
                            query = fe.extract(img)

                            find_one = image_features_collection.find_one({"feature":query.tolist()})

                            try:
                                dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
                                ids = np.argsort(dists)[:10]  # Top 20 results
                                scores = [ img_paths[id] for id in ids]

                                dists_score = [(dists[id], img_paths[id]) for id in ids]
                                print(f"results = {results}")

                                for id in ids:
                                    print(f"---------------------------->probdist{calculate_chi2_probability1(dists[id], degrees_of_freedom)}")
                                    if img_paths[id] in results :

                                        print(f"inside iff-{img_paths[id]} ---  {results[img_paths[id]]} value ends here {calculate_chi2_probability1(dists[id], degrees_of_freedom)}")
                                        results[img_paths[id]].append(float(calculate_chi2_probability1(dists[id], degrees_of_freedom)))
                                        print("------------inside if------------")
                                    else:
                                        print(f"----------inside else----------")
                                        results[img_paths[id]] = []+ [calculate_chi2_probability1(dists[id], degrees_of_freedom)]
                                        print(f"inside else----={img_paths[id]}  ----{calculate_chi2_probability1(dists[id], degrees_of_freedom)}  value ends here ")


                                final_score.extend([scores[0],scores[1]])
                                # final_score = final_score + list(set(scores)-list(final_score))
                            except Exception as e:
                                    print(e)

                            currentframe += 1
                    else:
                            break
            cam.release()
            cv2.destroyAllWindows()


            print("_________________________________________________________________________________")
            print(results)
            print("_________________________________________________________________________________")
            final_score = list(set(final_score))
            print(final_score)
            fs = gridfs.GridFS(database)
            images = []

            for keys in results:
                try:
                    results[keys] = math.floor(mean(results[keys]), 2)
                except Exception as e:
                    results[keys] = 0.0
            print(20*"{category}")
            if final_score :
                # return JsonResponse({'message': final_score})
                return JsonResponse({'message': results})
            else :
                return JsonResponse({'images': "no images found"})



        else:
            return JsonResponse({'error': 'No video file provided'}, status=400)


