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
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

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

    with open("sample_train_img_brightness.pkl", 'rb') as f_name:
        train_img = pickle.load(f_name)
        train_img /= 255.0

    with open("sample_train_label_brightness.pkl", 'rb') as f_name:
        train_label = pickle.load(f_name)
        train_label = tf.keras.utils.to_categorical(train_label.astype('int'))
        
    
    with open("sample_test_img_brightness.pkl", 'rb') as f_name:
        test_img = pickle.load(f_name)
        test_img /= 255.0

    with open("sample_test_label_brightness.pkl", 'rb') as f_name:
        test_label = pickle.load(f_name)
        test_label = tf.keras.utils.to_categorical(test_label.astype('int'))

    #Shuffling Images!

    permute_train = np.random.permutation(len(train_label))
    permute_test = np.random.permutation(len(test_label))

    train_img = train_img[permute_train]
    train_label = train_label[permute_train]

    test_img = test_img[permute_test]
    test_label = test_label[permute_test]

    return(train_img, test_img, train_label, test_label)


train_x, test_x, train_y, test_y = load_shuffled_data()
model_name = 'AlexNet_v3_Trial_4.h5'
#Evaluate Data
model = load_model(model_name)
model_summary = model.summary()
print(model_summary)
loss, acc = model.evaluate(test_x, test_y, verbose = 0)
print("Test, Accuracy: ", acc)
	
# predict probabilities for test set
yhat_probs = model.predict(test_x, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(test_x, verbose=0)

test_y = test_y[:,[1]]

yhat_probs = yhat_probs[:, 1]
print(yhat_classes)
print(yhat_probs)
print(test_y)
nl = len(test_y)
print(nl)


#performance metrics
accuracy = accuracy_score(test_y, yhat_classes)
print('Accuracy: %f' % accuracy)
precision = precision_score(test_y, yhat_classes)
print('Precision: %f' % precision)
recall = recall_score(test_y, yhat_classes)
print('Recall: %f' % recall)
f1 = f1_score(test_y, yhat_classes)
print('F1 score: %f' % f1)
auc = roc_auc_score(test_y, yhat_probs)
print('ROC AUC: %f' % auc)

