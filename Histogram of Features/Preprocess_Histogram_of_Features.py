import os
from scipy.ndimage import gaussian_filter
import numpy as np 
import pandas as pd 
import time 
from PIL import Image
import cv2
import pickle 
import json, codecs
import matplotlib.pyplot as plt

print("Program Began")

#defining dataset location
cur_dir = os.getcwd()
fold_dir = '/'.join((cur_dir, "no_augmentation_split(1)/no_augmentation_split/train/contains_car"))
car_names = sorted(os.listdir(fold_dir))

#importing images

img_cars = []

for i in range(0, 6554):
    file_name = "/".join((fold_dir, car_names[i]))
    img = np.array(Image.open(file_name).convert('LA'))
    img_cars.append(img[:,:,0])
    print(file_name)
    i += 1

print("Images Extracted!")

start_time = time.time()

fd_filter = []

r = 0

histogram = []
label=1  #for dataset with cars
#label=0 #for dataset without cars

for img in img_cars:
    img = cv2.resize(img, (512, 512))                          #resize to 512x512
    img_filtered = cv2.bilateralFilter(img,9,75,75)            #apply filter
    freak = cv2.xfeatures2d.FREAK_create()                     #creates freak features
    orb = cv2.ORB_create()                                     #creates orb features
    daisy = cv2.xfeatures2d.DAISY_create()                     #creates daisy features

    
    #manually denoting grid centroids as keypoints
    key_points = []
    for j in range(32,481,64):                                  #for 64 keypoints (64kp- 64, 100kp- 48, 225kp- 32, 841kp- 16)
        for i in range(32,481,64):
            key_point = cv2.KeyPoint(i, j, 1)
            key_points.append(key_point)
    kp, des = orb.compute(img_filtered, key_points)             #computing ORB features (daisy.compute for daisy, freak.compute for freak features)

    fd_filter.append(des)
    
    #going through each keypoint, plotting a histogram of the descriptor of the keypoint/grid cell, appending to an empty list patch_histogram. 
    patch_histogram = []
    p = 0
    for p in range(0,64):
        h = np.histogram(des[p], 256, (0, 256))
        patch_histogram.append(h[0])
        #print(p)
        p+=1

    patch_histogram = np.concatenate(patch_histogram)
    patch_histogram = np.array((label, patch_histogram))
    patch_histogram = patch_histogram.tolist()
    histogram.append(patch_histogram)
    print('\nImage Number: \n\n')
    print(r)
    print('\ndescriptor shape: \n\n')
    print(des.shape)
    r+=1


print("Process for {} images done in {}".format(len(img_cars), (time.time()- start_time)/60))

print('\n\n')
print('Shape of Histogram Array: \n\n')
print((histogram))
#print((final))

with open("cars_1_6554_train.txt", "wb") as file_car:
    pickle.dump(fd_filter, file_car)
with open("car_1_6554_train_histogram_2.txt", "wb") as file_car_histogram:
    pickle.dump(histogram, file_car_histogram)
