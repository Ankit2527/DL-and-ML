# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 17:07:35 2019

@author: Ankit
"""

import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import os
import matplotlib.pyplot as plt

train_cat_dir=os.path.join('D:/Datasets/cats_and_dogs_filtered/train/cats')
train_dog_dir=os.path.join('D:/Datasets/cats_and_dogs_filtered/train/dogs')
validation_cat_dir=os.path.join('D:/Datasets/cats_and_dogs_filtered/validation/cats')
validation_dog_dir=os.path.join('D:/Datasets/cats_and_dogs_filtered/validation/dogs')

print('total training cat images = ',len(os.listdir(train_cat_dir)))
print('total training dog images = ',len(os.listdir(train_dog_dir)))
print('total validation cat images = ',len(os.listdir(validation_cat_dir)))
print('total validation dog images = ',len(os.listdir(validation_dog_dir)))

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

train_generator=train_datagen.flow_from_directory('D:/Datasets/cats_and_dogs_filtered/train',
                                                  target_size=(150,150),
                                                  batch_size=20,
                                                  class_mode='binary')

validation_generator=validation_datagen.flow_from_directory('D:/Datasets/cats_and_dogs_filtered/validation',
                                                  target_size=(150,150),
                                                  batch_size=20,
                                                  class_mode='binary')

history=model.fit_generator(train_generator,
                            steps_per_epoch=100,
                            epochs=15,
                            validation_data=validation_generator,
                            validation_steps=50,
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