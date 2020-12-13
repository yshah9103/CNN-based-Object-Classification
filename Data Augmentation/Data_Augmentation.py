import cv2
import glob
import os
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import tensorflow as tf 
import random
import numpy as np

datagen3 = ImageDataGenerator(fill_mode='nearest', brightness_range=[1.75, 1.75])
datagen4 = ImageDataGenerator(fill_mode='nearest', horizontal_flip=True)
datagen = ImageDataGenerator(fill_mode='nearest', rotation_range= 45, zoom_range=[0.75, 0.75], brightness_range=[1.75,1.75])
datagen1 = ImageDataGenerator(fill_mode ='nearest', rotation_range=30)
#datagen2 = ImageDataGenerator(zoom_range=[0.75,0.75])
inputFolder = '/home/yash/Downloads/no_augmentation_split(1)/no_augmentation_split/test/no_car'
folderLen = len(inputFolder)
os.mkdir('Generated')
def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

def brightness(img, low, high):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value 
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def fill(img, h, w):
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img

i = 0
for img in glob.glob(inputFolder + '/*.jpg'):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image0 = cv2.flip(image, 1)
    image4 = cv2.flip(image, 0)
    image1 = rotation(image, 30)
    image2 = brightness(image, 1.75, 1.75)
    image3 = zoom(image, 0.75)
    pic_array = img_to_array(image)
    pic_array = pic_array.reshape((1,) + pic_array.shape)
    pic_array1 = img_to_array(image)
    pic_array1 = pic_array1.reshape((1,) + pic_array1.shape)
 
    count = 0
    for batch in datagen.flow(pic_array, batch_size=1,save_to_dir='/home/yash/repos/OIDv4_ToolKit/OID/Dataset/train/mixed', save_prefix='new', save_format='jpg'):
        count += 1
        if count == 1:
            break
    count = 0
    for batch in datagen3.flow(pic_array1, batch_size=1,save_to_dir='/home/yash/repos/OIDv4_ToolKit/OID/Dataset/train/brightness', save_prefix='new', save_format='jpg'):
        count += 1
        if count == 1:
            break

    aug_iter = datagen.flow(pic_array, batch_size=1)
    image = next(aug_iter)[0].astype('uint8')
    #aug_iter1 = datagen2.flow(pic_array3, batch_size=1)
    #image3 = next(aug_iter1)[0].astype('uint8')    
    print("generated")
    print(i)
    flipped = "Generated/flipped_%d.jpg"%i
    bright = "Generated/bright_%d.jpg"%i
    rotated = "Generated/rotated_%d.jpg"%i
    zoomed = "Generated/zoomed_%d.jpg"%i
    flippedv = "Generated/flippedv_%d.jpg"%i
    #imgResized = cv2.resize(image, (256, 256))
    cv2.imwrite(flipped, image0)
    cv2.imwrite(bright, image2)
    cv2.imwrite(rotated, image1)
    cv2.imwrite(zoomed, image3)
    cv2.imwrite(flippedv, image4)
    i+=1

    cv2.waitKey(30)
cv2.destroyAllWindows()


