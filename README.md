# CNN based Object Classification 
Computer Vision Project (CS/RBE 549 at WPI)

The goal of our project was to compare car recognition accuracy between convolutional networks and systems that used traditional computer vision methods. We also aimed to compare how their accuracy changed when they were tested with different augmented images.

### Approach


![Screenshot from 2021-09-20 11-55-23](https://user-images.githubusercontent.com/74123050/134033911-d8d39c21-cc29-46fb-9427-1cb41896b91d.png)


### Dataset
We obtained our cars dataset by using images from the ‘Stanford Cars Dataset’(Krause, 2013). We chose this dataset because it has a large number of images containing cars in a variety of settings and arbitrary orientations, which is desirable for training machine learning algorithms. The Stanford Cars Dataset consists of 8143 images of cars of various classes in different orientations and lighting conditions.

We obtained our non-car images from the ‘Open Images Dataset’ (Kuznetsova, 2020). This dataset was chosen because it has a variety of images from numerous classes that can be easily obtained. We obtained roughly 175 images each from 50 different classes.

To achieve the goal of our project, we firstly prepared a dataset of images. For classification purposes, this dataset consisted of two types of images: images that contained cars in different orientations, and images that did not contain any cars. We preprocessed these images by converting them to grayscale before using them.

![Screenshot from 2021-09-20 11-59-42](https://user-images.githubusercontent.com/74123050/134034625-4b4f7f10-70e4-49b2-9431-0f74f8cc79dc.png)


![Screenshot from 2021-09-20 11-59-52](https://user-images.githubusercontent.com/74123050/134034645-2726139e-a085-4ba2-affb-1a905dbbba64.png)

### Splitting Data
![Screenshot from 2021-09-20 12-04-42](https://user-images.githubusercontent.com/74123050/134035365-ffca97c1-15c8-4c54-8504-dcf227509668.png)

Before using the images in the dataset for our systems, we resized the images to 256x256 pixels to reduce the computation time and also to standardize our input. The images were also converted to grayscale, since grayscale images only have 1 channel of pixel intensities instead of 3 channels in color images, resulting in faster computation.



# Visual Bag of Words

Visual bags of words systems are based on bags of words systems, which count certain words in documents and use those words’ frequency as input for a machine learning classification algorithm. However, instead of counting words, visual bags of words systems attempt to count the frequency of certain image features.

![Screenshot from 2021-09-20 12-12-47](https://user-images.githubusercontent.com/74123050/134036725-4001c42d-18fb-42b0-ba00-4d42321bffb6.png)

The first step is to use a feature detector on every image in the dataset and gather all of the images’ feature descriptors. We attempted to use visual bags of words with two types of feature descriptors: ORB and FREAK.

The second step is to cluster all of the feature descriptors collected in the first step. Clustering is the process of separating data into groups so that similar data are in the same group.

![Screenshot from 2021-09-20 12-14-59](https://user-images.githubusercontent.com/74123050/134036953-0e02cce7-fb6a-41c0-b4e9-9e493c725771.png)

# Principle Component Analysis (PCA) and Support Vector Machine (SVM) based Classification

### Histogram of Features

Histograms of features is a system based on combined frequency histograms of image patch feature descriptions. This method creates a uniform feature description for each image, making it suitable as an input for a machine learning model. Finding an image description using histograms of image patches can theoretically describe the entire image without much information loss. The feature descriptors used for this method were FREAK, ORB, and HOG. 

![Screenshot from 2021-09-20 12-18-11](https://user-images.githubusercontent.com/74123050/134037441-4eff597d-99e1-4cba-adfc-4560d57a46f0.png)

First, the image is resized to decrease computation time. After some experiments, we decided to use 512x512 images as this particular resolution could form a uniform (nxn) grid having square grid cells. The 512x512 images were then divided into square grids of 64x64 pixels each resulting in 8x8 (64) grids in total for each image. After the grids were formed, each grid cell’s centroid was assigned to be a keypoint which would then be used for feature description.


![Screenshot from 2021-09-20 12-19-58](https://user-images.githubusercontent.com/74123050/134037724-a7594bd6-630d-4966-bcbb-334d76e48e4f.png)

The image description was run through a Principal Component Analysis (PCA). PCA essentially reduces the size of the image description to a desired number of components. The image descriptors are ordered based on their influence on the overall image description and the  image descriptors within a certain variance range were  selected. We tested this implementation for 500, 800, 1000, and 5000 components.

Finally, the resulting image descriptions would be used to train a support vector machine(SVM), which is a machine learning algorithm for object classification. After labeling the training images appropriately based on their class (car vs non-car), the SVM is used to classify the images as car or non-car images.

### Histogram of Gradients

Histogram of Gradients(HOG) is a global descriptor that is used to describe an image based on edge orientations. It is an effective method of describing an image since it preserves edge information very well. For an image of size 512x512, the HOG feature descriptor shape for an image is (1, 8192).


![Screenshot from 2021-09-20 12-25-02](https://user-images.githubusercontent.com/74123050/134038703-b1465c0e-e11c-4bf4-acba-d76d499f00b8.png)

The resultant feature description after PCA is used to train the machine learning classifier which in this case was an SVM.

# Convolutional Neural Network (CNN) based Classification

The primary reason we implemented a version of a convolutional neural network was to compare the effectiveness of hand-crafted feature descriptors to the performance of the learned features from convolutional neural networks. We also wished to understand the generalizability of the “crafted” and “learned” features  when testing them with augmented images. For the purpose of comparison, our team decided to implement AlexNet (Krizhevsky, 2012) from scratch instead of using a pre-trained model. We did this to interpret the generalizability of a particular type of object’s feature descriptors.
With respect to the implementation of AlexNet (Krizhevsky, 2012), there are a total of 11 layers. Typically, the convolutional layers are a variety of convolutional layers with different filters & window sizes, pooling, and batch normalization. After several convolutional layers, there is a flatten layer that converts the two-dimensional matrix into a one-dimensional array to be used as an input for fully connected layers used for classification. To implement a more scalable model given our resources, we chose to use three convolutional layers of different filter sizes - 96 -> 256 -> 128 instead of the filters used in AlexNet. The fully connected layers were the same as the ones in AlexNet, and the output layer had two nodes in order to get a prediction probability for both classes given an image.

# Data Augmentation

We applied each of the following augmentations, creating a new image after each augmentation.

Increasing brightness by 1.75 times.
Horizontally flipping the image.
Zooming into the image by 1.25 times.
Rotating the image by 0 to 30 degrees.
Combining zoom by 1.25 times, rotation (0-30 degrees), and brightness by 1.75 times.

![Screenshot from 2021-09-20 12-33-07](https://user-images.githubusercontent.com/74123050/134039586-00faa9cb-317b-4e27-a263-2753cb3bfc45.png)


# Results

The accuracy metrics that we used to evaluate our car recognition systems were precision, recall, F1 score, and ROC AUC score.


![Screenshot from 2021-09-20 12-35-59](https://user-images.githubusercontent.com/74123050/134040084-31aa234e-fd9d-4448-8311-202dd483f04c.png)

![Screenshot from 2021-09-20 12-37-22](https://user-images.githubusercontent.com/74123050/134040179-800dc1d7-8c67-4164-929f-2e07b1c6b20d.png)

