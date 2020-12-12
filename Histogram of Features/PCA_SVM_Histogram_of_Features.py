import pandas as pd
import numpy as np 
import os
import time
import pickle
from sklearn.decomposition import LatentDirichletAllocation as LDA, PCA
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix, classification_report


#Defining path to Histogram Data (ORB/Freak/HoG)
folder_path = os.getcwd()
hist_path = "/".join((folder_path, "ORB_HISTOGRAM", "Histograms"))
train_car_path = "/".join((hist_path, "Car", "Train"))
train_noise_path = "/".join((hist_path, "nocar", "Train"))
valid_car_path = "/".join((hist_path, "Car", "Val"))
valid_noise_path = "/".join((hist_path, "nocar", "Val"))

train_car_files = os.listdir(train_car_path)
train_car_hist_path = "/".join((train_car_path, train_car_files[0]))
train_noise_files = os.listdir(train_noise_path)
train_noise_hist_path = "/".join((train_noise_path, train_noise_files[0]))

valid_car_files = os.listdir(valid_car_path)
valid_car_hist_path = "/".join((valid_car_path, valid_car_files[0]))
valid_noise_files = os.listdir(valid_noise_path)
valid_noise_hist_path = "/".join((valid_noise_path, valid_noise_files[0]))

#Loading the Histograms
train_car_hist = pd.DataFrame(pickle.load(open(train_car_hist_path, 'rb')))
train_noise_hist = pd.DataFrame(pickle.load(open(train_noise_hist_path, 'rb')))

valid_car_hist = pd.DataFrame(pickle.load(open(valid_car_hist_path, 'rb')))
valid_noise_hist = pd.DataFrame(pickle.load(open(valid_noise_hist_path, 'rb')))

#Separating Image Histograms and Labels (Training Set)
X_train_car = np.vstack(train_car_hist[1].values.tolist())
Y_train_car = train_car_hist[0]

X_train_noise= np.vstack(train_noise_hist[1].values.tolist())
Y_train_noise= train_noise_hist[0]

X_train = np.concatenate((X_train_car, X_train_noise))
Y_train = np.concatenate((Y_train_car, Y_train_noise))

print(np.shape(X_train_car))
print(np.shape(X_train_noise))

print(np.shape(X_train))

#Shuffling to avoid learning skew
index = np.random.permutation(len(X_train))
X_train_shuffled = X_train[index]
Y_train_shuffled = Y_train[index]


#Separating Image Histograms and Labels (Validation Set)
X_valid_car = np.vstack(valid_car_hist[1].values.tolist())
Y_valid_car = valid_car_hist[0]
X_valid_noise = np.vstack(valid_noise_hist[1].values.tolist())
Y_valid_noise = valid_noise_hist[0]


X_valid = np.concatenate((X_valid_car, X_valid_noise))
Y_valid = np.concatenate((Y_valid_car, Y_valid_noise))

#Shuffling to avoid learning skew
index = np.random.permutation(len(X_valid))
X_valid_shuffled = X_valid[index]
Y_valid_shuffled = Y_valid[index]

#Performing PCA
print(X_train_shuffled)
pca_50 = PCA(n_components = 50)
print('done')
X_train_pca_50 = pca_50.fit_transform(X_train_shuffled)
print(pca_50.explained_variance_[-1])
X_valid_pca_50 = pca_50.transform(X_valid_shuffled)
print(pca_50.explained_variance_[-1])

#Predicting using SVM
model_svm_50 = SVC()
#model_svm_50.fit(X_train_shuffled, Y_train_shuffled)
model_svm_50.fit(X_train_pca, Y_train)


Y_predict_svm_50 = model_svm_50.predict(X = X_valid_shuffled)

#Displaying metrics based on prediction
f1_50 = f1_score(y_true = Y_valid_shuffled, y_pred = Y_predict_svm_50)
print(f1_50)