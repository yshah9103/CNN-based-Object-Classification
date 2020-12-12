import os
from scipy.ndimage import gaussian_filter
import numpy as np 
import pandas as pd 
from skimage.feature import hog
import time
from skimage import data, exposure
from PIL import Image
import cv2
import pickle 
import json, codecs
kernel = np.ones((5,5),np.uint8)
print("Program Began")
cur_dir = os.getcwd()
fold_dir = '/'.join((cur_dir, "Carsaug/test_cars"))
car_names = sorted(os.listdir(fold_dir))

img_cars = []

for i in range(0, 783):
    file_name = "/".join((fold_dir, car_names[i]))
    img = np.array(Image.open(file_name).convert('LA'))
    #img_cars.append(img[:,:,0])
    img_cars.append(img)
    print(file_name)
    i += 1

print("Images Extracted!")

start_time = time.time()
label = 1
#Extracting Gauss + HoG Features
start_time = time.time()
fd_hog = []
r = 0
for img in img_cars:
    img = cv2.resize(img,(512,512))
    img = cv2.blur(image, (25,25))
    img_filtered = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    fd, hog_image = hog(img_filtered, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)
    fd = np.array((label, fd))
    fd = fd.tolist()
    fd_hog.append(fd)
    print (r)
    r+=1

print("Process for {} images done in {}".format(len(img_cars), (time.time()- start_time)/60))
print(np.shape(fd_hog))

with open("cars_1_783_hog_label_testmixed.txt", "wb") as file_car:
    pickle.dump(fd_hog, file_car)