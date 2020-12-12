import numpy as np
import pandas as pd 
import os
import time
import cv2
from PIL import Image
import tensorflow as tf
from random import randrange
import pickle
import json

def extract_resize_img(folder_path, dataset_type, label_name):
    """
    Extract images in grayscale and resize them. 
    Return back the extracted images with associated label.
    """
    dataset_path = "/".join((folder_path, dataset_type))
    img_dir = os.listdir(dataset_path)
    img_data = []
    for img_name in img_dir:
        img_path = "/".join((dataset_path, img_name))
        #print("Path: ", img_path)
        img_grey = np.array(Image.open(img_path).convert('LA'))
        img_grey = tf.image.resize(img_grey, (256, 256), preserve_aspect_ratio=False).numpy()
        #img_grey = img_grey.reshape(256, 256, 1)
        #img_data.append([img_grey[:, :, 0]])

        img_data.append([img_grey[:, :, 0]])

    return(np.array(img_data))

def random_shuffle(img, label = 1):
    label_mtx = label* np.shape(img)[0]
    permutation = np.random.permutation(np.shape(label))
    img = img[permutation]
    label = img[permutation]

folder_path = os.getcwd()
car_path = "/".join((folder_path, "Cars"))
noise_path = "/".join((folder_path, "Noise"))
car_path1 = "/".join((folder_path, "Carsaug"))
noise_path1 = "/".join((folder_path, "Noiseaug"))

start_time = time.time()
car_img_train = extract_resize_img(folder_path = car_path, dataset_type = 'train', label_name = 1)
car_img_train = car_img_train.reshape(car_img_train.shape[0], 256, 256, 1)

car_img_test = extract_resize_img(folder_path = car_path1, dataset_type = 'test_cars', label_name = 1)
car_img_test = car_img_test.reshape(car_img_test.shape[0], 256, 256, 1)

car_img_val = extract_resize_img(folder_path = car_path1, dataset_type = 'val_cars', label_name = 1)
car_img_val = car_img_val.reshape(car_img_val.shape[0], 256, 256, 1)

noise_img_train = extract_resize_img(folder_path = noise_path, dataset_type = 'train', label_name = 0)
noise_img_train = noise_img_train.reshape(noise_img_train.shape[0], 256, 256, 1)

noise_img_test = extract_resize_img(folder_path = noise_path1, dataset_type = 'test_nocar', label_name = 0)
noise_img_test = noise_img_test.reshape(noise_img_test.shape[0], 256, 256, 1)

noise_img_val = extract_resize_img(folder_path = noise_path1, dataset_type = 'val_nocars', label_name = 0)
noise_img_val = noise_img_val.reshape(noise_img_val.shape[0], 256, 256, 1)

print("Time Taken: ", time.time() - start_time)

label_car_train = [1] * car_img_train.shape[0]
label_noise_train = [0] * noise_img_train.shape[0]
train_img = np.concatenate([car_img_train, noise_img_train])
train_label = np.concatenate([label_car_train, label_noise_train])

permute_train = np.random.permutation(len(train_label))

train_img   = train_img[permute_train]
train_label = train_label[permute_train]

label_car_test= [1] * car_img_test.shape[0]
label_noise_test = [0] * noise_img_test.shape[0]
test_img = np.concatenate([car_img_test, noise_img_test])
test_label = np.concatenate([label_car_test, label_noise_test])

permute_test = np.random.permutation(len(test_label))

test_img   = test_img[permute_test]
test_label = test_label[permute_test]

label_car_val= [1] * car_img_val.shape[0]
label_noise_val = [0] * noise_img_val.shape[0]
val_img = np.concatenate([car_img_val, noise_img_val])
val_label = np.concatenate([label_car_val, label_noise_val])

permute_val = np.random.permutation(len(val_label))

val_img   = val_img[permute_val]
val_label = val_label[permute_val]

test_label

with open("sample_train_img_brightness.pkl", "wb") as f_name:
    pickle.dump(train_img, f_name)

with open("sample_train_label_brightness.pkl", "wb") as f_name:
    pickle.dump(train_label, f_name)

with open("sample_test_img_brightness.pkl", "wb") as f_name:
    pickle.dump(test_img, f_name)

with open("sample_test_label_brightness.pkl", "wb") as f_name:
    pickle.dump(test_label, f_name)

with open("sample_val_img_brightness.pkl", "wb") as f_name:
    pickle.dump(val_img, f_name)

with open("sample_val_label_brightness.pkl", "wb") as f_name:
    pickle.dump(val_label, f_name)

#with open("sample_test_img_brightness.pkl", "rb") as f_name:
#    train_sample = pickle.load(f_name)

#with open("sample_train_label_brightness.pkl", "rb") as f_name:
#    train_label_sample = pickle.load(f_name)

