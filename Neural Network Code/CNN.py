import tensorflow as tf
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

import cv2
from PIL import Image
import tensorflow as tf
from random import randrange
import pickle
import json
import tensorflow.keras as keras
import os
import sys
import time

def load_shuffled_data():
    """
    Load dataset from pickle file and split features and labels
    returns (X_train, X_test, y_train, y_test)
    """
    start_time = time.time()

    with open("sample_train_img.pkl", 'rb') as f_name:
        train_img = pickle.load(f_name)
        train_img /= 255.0

    with open("sample_train_label.pkl", 'rb') as f_name:
        train_label = pickle.load(f_name)
        train_label = tf.keras.utils.to_categorical(train_label.astype('int'))
        
    
    with open("sample_test_img.pkl", 'rb') as f_name:
        test_img = pickle.load(f_name)
        test_img /= 255.0

    with open("sample_test_label.pkl", 'rb') as f_name:
        test_label = pickle.load(f_name)
        test_label = tf.keras.utils.to_categorical(test_label.astype('int'))

    with open("sample_val_img.pkl", 'rb') as f_name:
        val_img = pickle.load(f_name)
        val_img /= 255.0

    with open("sample_val_label.pkl", 'rb') as f_name:
        val_label = pickle.load(f_name)
        val_label = tf.keras.utils.to_categorical(val_label.astype('int'))

    #Shuffling Images!

    permute_train = np.random.permutation(len(train_label))
    permute_test = np.random.permutation(len(test_label))
    permute_val = np.random.permutation(len(val_label))

    train_img = train_img[permute_train]
    train_label = train_label[permute_train]

    test_img = test_img[permute_test]
    test_label = test_label[permute_test]

    val_img = val_img[permute_val]
    val_label = val_label[permute_val]

    return(train_img, test_img, val_img, train_label, test_label, val_label)

def CNN_model(image_height = 256, image_width = 256, image_channel=1, class_count = 2):
    
    model = Sequential()
    #Layer 1
    model.add(Conv2D(   filters = 96, 
                        kernel_size=(11,11), 
                        strides = 4, 
                        activation='relu', 
                        kernel_initializer='he_uniform', 
                        padding = "same",
                        input_shape = (image_height, image_width, image_channel)))

    model.add(BatchNormalization())

    #Taking the maximum in that subregion using MaxPooling
    model.add(MaxPool2D(    pool_size = (3, 3),
                            strides = 2))
    #he_uniform: Draw samples from uniform distributions within range of limits created with units


    #Layer 2
    model.add(Conv2D(   filters = 256, 
                        kernel_size=(5,5), 
                        strides = 1,
                        activation='relu', 
                        kernel_initializer='he_uniform'))

    model.add(BatchNormalization())
    model.add(MaxPool2D(    pool_size = (3, 3),
                            strides = 2))

    #Layer 3: 
    model.add(Conv2D(   filters = 128, 
                        kernel_size=(3,3), 
                        strides = 1, 
                        activation='relu', 
                        kernel_initializer='he_uniform'))

    model.add(BatchNormalization())
    model.add(MaxPool2D(    pool_size=(3,3),
                            strides = 2))
    
    #Layer 4
    model.add(Flatten())

    #Layer 5:
    model.add(Dense(    units = 4096, 
                        activation='relu',              
                        kernel_initializer='he_uniform'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))   

    #Layer 6
    model.add(Dense(    units = 4096, 
                        activation='relu', 
                        kernel_initializer='he_uniform'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))   

    #Layer 7
    model.add(Dense(    units = 2, 
                        activation='softmax'))

    #Compiling:
    opt = SGD(lr = 0.001, momentum = 0.9)
    model.compile(  optimizer=opt, 
                    loss='categorical_crossentropy',
                    metrics = ['accuracy'])

    return model

cnn_model = CNN_model()
train_x, test_x, val_x, train_y, test_y, val_y = load_shuffled_data()

np.shape(train_x)

model_name = 'AlexNet_v3_Trial_4.h5'

#Fit Model!
es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=10)
mc = ModelCheckpoint(model_name, monitor='val_accuracy', mode='max', verbose=2, save_best_only=True)
history = cnn_model.fit(train_x, train_y, epochs = 10, batch_size = 32, validation_data=(val_x,val_y), verbose = 1, callbacks = [es, mc], shuffle=True)

#Evaluate Data
model = load_model(model_name)
loss, acc = model.evaluate(test_x, test_y, verbose = 0)
print("Test, Accuracy: ", acc)

