"""
  File name:    model.py
  Author:       Christopher Cabrera
  Date created: 8/2/2017
  Desription:   Implement, train, and save the NVIDIA neural net.
"""

import cv2
import csv
import random
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# Read data needed from driving log file to 
# be used by kera input generators.

correction = 0.2
samples = []
with open('./data/driving_log.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append((line[0].split('/')[-1], float(line[3])))
        samples.append((line[1].split('/')[-1], float(line[3])+correction))
        samples.append((line[2].split('/')[-1], float(line[3])-correction))
    
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# Define generators for the creation of the
# data for training and validation.

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image = cv2.cvtColor(cv2.imread('./data/IMG/' + batch_sample[0]), cv2.COLOR_BGR2RGB)
                angle = float(batch_sample[1])
                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=256)
validation_generator = generator(validation_samples, batch_size=256)


# Use keras to implement the Nvidia neural net for
# behavioral cloning in autonomouse vehicles and train
# net using the above data generators.

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))

model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, 
                    steps_per_epoch=len(train_samples)/256, 
                    validation_data=validation_generator, 
                    validation_steps=len(validation_samples)/256, 
                    epochs=5)

model.save('model.h5')