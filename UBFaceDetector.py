'''
All of your implementation should be in this file.
'''
'''
This is the only .py file you need to submit. 
'''
'''
    Please do not use cv2.imwrite() and cv2.imshow() in this function.
    If you want to show an image for debugging, please use show_image() function in helper.py.
    Please do not save any intermediate files in your final submission.
'''
from helper import show_image

import cv2

import numpy as np
import os
import sys
import re
import face_recognition

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''


def detect_faces(input_path: str) -> dict:

    result_list = []
    '''
    Your implementation.
    '''
    print(input_path)
    # print(str)
    # print(dict)
    directory_path = input_path
    images = os.listdir(directory_path)
    # images.sort(key=lambda a: int(re.sub('\D','', a)))
    # print(images)
    number_of_images = len(images)
    print(number_of_images)
   

    cv2_dir = os.path.dirname(cv2.__file__)

    xml_dir = os.path.join(cv2_dir, 'data')

    # face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haracascade_frontalface_alt.xml')
    face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'data/haarcascade/haarcascade_frontalface_default.xml')
    # imgs = []


    for ipath in range(1,number_of_images+1):
        imgpath = './'+input_path+'/img_'+str(ipath)+'.jpg'
        print(imgpath)
        img = cv2.imread(imgpath)
        # cv2.imshow('images', img)
        gray_images = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(gray_images)
        faces_detected = face_detection.detectMultiScale(gray_images,1.15,3)
        # print(faces_detected)
        # print(ipath)
        for (x, y, w, h) in faces_detection:
            # a single element for JSON in a particular format
            element = {"iname": faces_detected[ipath - 1],
                       "bbox": [int(x), int(y), int(w), int(h)]}
            result_list.append(element)

    return result_list


# output_json = "results.json"
# with open(output_json, 'w') as f:
#     json.dump(result_list, f)
'''
K: number of clusters
'''
def cluster_faces(input_path: str, K: int) -> dict:
    result_list = []
    '''
    Your implementation.
    '''
    return result_list


'''
If you want to write your implementation in multiple functions, you can write them here. 
But remember the above 2 functions are the only functions that will be called by FaceCluster.py and FaceDetector.py.
'''

"""
Your implementation of other functions (if needed).
"""
