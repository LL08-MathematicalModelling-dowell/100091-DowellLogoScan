#backup code for videoframesstroring
            # cam = cv2.VideoCapture(video_path)
            # while(True):

            #     ret,frame = cam.read()
            #     count_frame_buffer = count_frame_buffer  + 1
            #     if count_frame_buffer%2 ==0:
            #         print(f"---------->{count_frame_buffer}<-----------")
            #         if ret:
            #                 # name = './data/frame' + str(currentframe) + '.jpg'
            #                 name = f"{upload_image_path}/currentframe.jpg"
            #                 # upload_image_path =os.path.join(upload_image_path, name)
            #                 cv2.imwrite(name, frame)
            #                 img = Image.open(name)
            #                 query = fe.extract(img)

            #                 find_one = image_features_collection.find_one({"feature":query.tolist()})

            #                 try:
            #                     if not find_one :
            #                         fs = gridfs.GridFS(database)
            #                         with open(name,'rb') as f:
            #                             contents = f.read()
            #                             video_id = fs.put(contents,filename ="tempname",
            #                                  **{
            #                                 'category': category,
            #                                 'product': product,
            #                                 'brand': brand,
            #                                 'flag': 'user_upload'
            #                             }
            #                             )
            #                             data ={
            #                             'id':str(video_id),
            #                             'feature':query.tolist(),
            #                              'category': category,
            #                             'product': product,
            #                             'brand': brand,
            #                             'flag': 'user_upload'
            #                             }
            #                             user_image_features_collection.insert_one(data)

            #                 except Exception as e:
            #                         print(e)

            #                 currentframe += 1
            #         else:
            #                 break
            # cam.release()
            # cv2.destroyAllWindows()
