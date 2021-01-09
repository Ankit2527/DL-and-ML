# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 18:38:25 2019

@author: Ankit
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import os
import matplotlib.pyplot as plt
from shutil import copyfile
import random

print(tf.__version__)

train_dir=os.path.join('D:/Datasets/CATS_DOGS/CAT')
test_dir=os.path.join('D:/Datasets/CATS_DOGS/DOG')

print('total training  images = ',len(os.listdir(train_dir)))
print('total test images = ',len(os.listdir(test_dir)))

try:
    os.mkdir('D:/Datasets/CATS_DOGS')
    os.mkdir('D:/Datasets/CATS_DOGS/train')
    os.mkdir('D:/Datasets/CATS_DOGS/test')
    os.mkdir('D:/Datasets/CATS_DOGS/train/cat')
    os.mkdir('D:/Datasets/CATS_DOGS/train/dog')
    os.mkdir('D:/Datasets/CATS_DOGS/test/cat')
    os.mkdir('D:/Datasets/CATS_DOGS/test/dog')
except OSError:
    pass

def split_data(SOURCE,TRAINING,TESTING,SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file=SOURCE+filename
        if os.path.getsize(file) >0:
            files.append(filename)
        else:
            print(filename +" is zero length, so ignoring.")

    training_length=int(len(files)*SPLIT_SIZE)
    testing_length=int(len(files)-training_length)
    shuffled_set=random.sample(files,len(files))
    training_set=shuffled_set[0:training_length]
    testing_set=shuffled_set[-testing_length:]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)


CAT_SOURCE_DIR="D:/Datasets/CATS_DOGS/CAT/"
TRAINING_CATS_DIR="D:/Datasets/CATS_DOGS/train/cat/"
TESTING_CATS_DIR="D:/Datasets/CATS_DOGS/test/cat/"
DOG_SOURCE_DIR="D:/Datasets/CATS_DOGS/DOG/"
TRAINING_DOGS_DIR="D:/Datasets/CATS_DOGS/train/dog/"
TESTING_DOGS_DIR="D:/Datasets/CATS_DOGS/test/dog/"

split_size=0.9
split_data(CAT_SOURCE_DIR,TRAINING_CATS_DIR,TESTING_CATS_DIR,split_size)
split_data(DOG_SOURCE_DIR,TRAINING_DOGS_DIR,TESTING_DOGS_DIR,split_size)

print(len(os.listdir('D:/Datasets/CATS_DOGS/train/cat')))
print(len(os.listdir('D:/Datasets/CATS_DOGS/train/dog')))
print(len(os.listdir('D:/Datasets/CATS_DOGS/test/cat')))
print(len(os.listdir('D:/Datasets/CATS_DOGS/test/dog')))



model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
    ])

model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['acc'])

train_datagen=ImageDataGenerator(rescale=1./255)
validation_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory('D:/Datasets/CATS_DOGS/train',
                                                  target_size=(150,150),
                                                  batch_size=100,
                                                  class_mode='binary')

validation_generator=validation_datagen.flow_from_directory('D:/Datasets/CATS_DOGS/test',
                                                  target_size=(150,150),
                                                  batch_size=100,
                                                  class_mode='binary')

history=model.fit_generator(train_generator,
                            epochs=15,
                            validation_data=validation_generator,
                            verbose=2)

acc=history.history['acc']
val_acc=history.history['val_acc']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))

plt.plot(epochs,acc)
plt.plot(epochs,val_acc)
plt.title('Training and Validation accuracy')

plt.figure()

plt.plot(epochs,loss)
plt.plot(epochs,val_loss)
plt.title('training and validation loss')