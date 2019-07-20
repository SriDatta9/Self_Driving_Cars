
!git clone https://github.com/rslim087a/track  

!ls track

!pip3 install imgaug

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras. models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dropout,Dense
from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import ntpath
import random
import os

datadir = 'track'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(datadir,'driving_log.csv'), names=columns)
pd.set_option('display.max_colwidth', -1)
data.head()

def path_leaf(path):
  head,tail = ntpath.split(path)
  return tail

data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)
data.head()

##### visualizing the distribution of steering angles to no. of samples
num_bins = 25
samples_per_bin = 400
hist, bins = np.histogram(data['steering'], num_bins)
center = (bins[:-1]+ bins[1:])/2
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
plt.bar(center,hist,width = 0.05)

print('total_data:', len(data))
remove_list = []
for j in range(num_bins):
  list_ = []
  for i in range(len(data['steering'])):
    if data['steering'][i]>=bins[j] and data['steering'][i]<=bins[j+1] :
      list_.append(i)
  list_ = shuffle(list_)
  list_ = list_[samples_per_bin:]
  remove_list.extend(list_)

print('removed', len(remove_list))
data.drop(data.index[remove_list],inplace=True)
print('remaining', len(data))

########### As most of track is straight, limiting the no. of samples ber bin to 400 to reduce overfitting
hist,_ = np.histogram(data['steering'], num_bins)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
plt.bar(center,hist,width = 0.05)

def load_img_steering(datadir, df):
  image_path = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
    image_path.append(os.path.join(datadir, center.strip()))
    steering.append(float(indexed_data[3]))

    # left image append(adding 0.15 to left steering angle)
    image_path.append(os.path.join(datadir,left.strip()))
    steering.append(float(indexed_data[3])+0.15)

    # right image append(subtracting 0.15 to right steering angle)
    image_path.append(os.path.join(datadir,right.strip()))
    steering.append(float(indexed_data[3])-0.15)

  image_paths = np.asarray(image_path)
  steerings = np.asarray(steering)
  return image_paths, steerings

image_paths, steerings = load_img_steering(datadir +'/IMG', data)
X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state = 6)
print('Training Samples : {}\n Valid Samples : {}'.format(len(X_train), len(X_valid)))

fig, axs = plt.subplots(1,2, figsize=(12,4))
axs[0].hist(y_train, bins = num_bins, width = 0.05, color = 'blue')
axs[0].set_title('Training set')
axs[1].hist(y_train, bins = num_bins, width = 0.05, color = 'red')
axs[1].set_title('Validation set')

############# Data Augmentation
####### zooming image
def zoom(image):
  zoom = iaa.Affine(scale=(1,1.3))
  image = zoom.augment_image(image)
  return image

image = image_paths[random.randint(0,1000)]
original_image = mpimg.imread(image)
zoomed_image = zoom(original_image)

fig, axs = plt.subplots(1,2, figsize = (15,10))
fig.tight_layout()

axs[0].imshow(original_image)
axs[0].set_title("Original_image")
axs[1].imshow(zoomed_image)
axs[1].set_title("Zoomed_image")

####### pan image
def pan(image):
  pan = iaa.Affine(translate_percent = {'x ': (-0.1, 0.1), 'y' : (-0.1,0.1)})
  image = pan.augment_image(image)
  return image

image = image_paths[random.randint(0,1000)]
original_image = mpimg.imread(image)
pan_image = pan(original_image)

fig, axs = plt.subplots(1,2, figsize = (15,10))
fig.tight_layout()

axs[0].imshow(original_image)
axs[0].set_title("Original_image")
axs[1].imshow(pan_image)
axs[1].set_title("pan_image")

####### random brightness
def img_random_brightness(image):
  brightness = iaa.Multiply((0.2,1.2))
  image = brightness.augment_image(image)
  return image

image = image_paths[random.randint(0,1000)]
original_image = mpimg.imread(image)
bright_image = img_random_brightness(original_image)

fig, axs = plt.subplots(1,2, figsize = (15,10))
fig.tight_layout()

axs[0].imshow(original_image)
axs[0].set_title("Original_image")
axs[1].imshow(bright_image)
axs[1].set_title("bright_image")

########## flip image
def img_random_flip(image, steering_angle):
  image = cv2.flip(image, 1)
  steering_angle = -steering_angle
  return image, steering_angle

random_index = random.randint(0,1000)
image = image_paths[random_index]
steering_angle = steerings[random_index]
original_image = mpimg.imread(image)
flipped_image , flipped_steering_angle= img_random_flip(original_image, steering_angle)

fig, axs = plt.subplots(1,2, figsize = (15,10))
fig.tight_layout()

axs[0].imshow(original_image)
axs[0].set_title("Original_image")
axs[1].imshow(flipped_image)
axs[1].set_title("flipped_image")

def random_augment(image, steering_angle):
    image = mpimg.imread(image)
    if np.random.rand() < 0.5:
      image = pan(image)
    if np.random.rand() < 0.5:
      image = zoom(image)
    if np.random.rand() < 0.5:
      image = img_random_brightness(image)
    if np.random.rand() < 0.5:
      image, steering_angle = img_random_flip(image, steering_angle)

    return image, steering_angle

ncol = 2
nrow = 10

fig, axs=  plt.subplots(nrow,ncol, figsize=(15,50))
fig.tight_layout()

for i in range(10):
  rand_num = random.randint(0,len(image_paths)-1)
  random_image = image_paths[rand_num]
  random_steering = steerings[rand_num]

  original_image = mpimg.imread(random_image)
  augmented_image, steering = random_augment(random_image, random_steering)

  axs[i][0].imshow(original_image)
  axs[i][0].set_title('Original Image')

  axs[i][1].imshow(augmented_image)
  axs[i][1].set_title('Augmented Image')

########### preprocess image
def img_preprocess(image):
  image = image[60:135, :, :]
  image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
  image = cv2.GaussianBlur(image, (3,3),0)
  image = cv2.resize(image,(200,66))
  image = image/255
  return image

image = image_paths[100]
original_image = mpimg.imread(image)
preprocessed_image = img_preprocess(original_image)

fig, axs = plt.subplots(1,2, figsize = (15,10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('Original image')
axs[1].imshow(preprocessed_image)
axs[1].set_title("processed image")

############### batch generator
def batch_generator(image_paths, steering_angle, batch_size, istraining):
  while True:
    batch_img = []
    batch_steering = []
    for i in range(0, batch_size):
      random_index = random.randint(0, len(image_paths)-1)

      if istraining :
        im, steering = random_augment(image_paths[random_index], steering_angle[random_index])

      else:
        im = mpimg.imread(image_paths[random_index])
        steering = steering_angle[random_index]

      im = img_preprocess(im)
      batch_img.append(im)
      batch_steering.append(steering)

    yield(np.asarray(batch_img), np.asarray(batch_steering))

######## using nvidia model (taken from nvidia "end to end learning for self driving cars" paper)
def nvidia_model():
  model = Sequential()
  model.add(Convolution2D(24,(5,5), subsample = (2,2), input_shape = (66,200,3), activation = 'elu'))
  model.add(Convolution2D(36,(5,5), subsample = (2,2), activation = 'elu'))
  model.add(Convolution2D(48,(5,5), subsample = (2,2), activation = 'elu'))
  model.add(Convolution2D(64,(3,3), activation = 'elu'))
  model.add(Convolution2D(64,(3,3), activation = 'elu'))
  #model.add(Dropout(0.5))

  model.add(Flatten())

  model.add(Dense(100,activation = 'elu'))
  #model.add(Dropout(0.5))

  model.add(Dense(50,activation = 'elu'))
  #model.add(Dropout(0.5))

  model.add(Dense(10,activation = 'elu'))
  #model.add(Dropout(0.5))

  model.add(Dense(1))
  model.compile(Adam(lr=0.0001), loss = 'mse')
  return model

model = nvidia_model()
print(model.summary())

#### no. of epochs = 10
h = model.fit_generator(batch_generator(X_train, y_train, 100, 1),
                                  steps_per_epoch=300,
                                  epochs=10,
                                  validation_data=batch_generator(X_valid, y_valid, 100, 0),
                                  validation_steps=200,
                                  verbose=1,
                                  shuffle = 1)

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epochs')

###### saving model.h5
model.save('model.h5')

from google.colab import files
files.download('model.h5')
