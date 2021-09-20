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

# Histogram of Featues

Histograms of features is a system based on combined frequency histograms of image patch feature descriptions. This method creates a uniform feature description for each image, making it suitable as an input for a machine learning model. Finding an image description using histograms of image patches can theoretically describe the entire image without much information loss. The feature descriptors used for this method were FREAK, ORB, and HOG. 

![Screenshot from 2021-09-20 12-18-11](https://user-images.githubusercontent.com/74123050/134037441-4eff597d-99e1-4cba-adfc-4560d57a46f0.png)

First, the image is resized to decrease computation time. After some experiments, we decided to use 512x512 images as this particular resolution could form a uniform (nxn) grid having square grid cells. The 512x512 images were then divided into square grids of 64x64 pixels each resulting in 8x8 (64) grids in total for each image. After the grids were formed, each grid cell’s centroid was assigned to be a keypoint which would then be used for feature description.


![Screenshot from 2021-09-20 12-19-58](https://user-images.githubusercontent.com/74123050/134037724-a7594bd6-630d-4966-bcbb-334d76e48e4f.png)
