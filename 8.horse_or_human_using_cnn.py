# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 00:35:46 2019

@author: Ankit
"""

import tensorflow as tf
from tensorflow import keras
import os


train_horse_dir = os.path.join('D:/horse-or-human/train/horses')
train_human_dir = os.path.join('D:/horse-or-human/train/humans')
validation_horse_dir = os.path.join('D:/horse-or-human/validation/horses')
validation_human_dir = os.path.join('D:/horse-or-human/validation/humans')

train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

validation_horse_hames = os.listdir(validation_horse_dir)
print(validation_horse_hames[:10])

validation_human_names = os.listdir(validation_human_dir)
print(validation_human_names[:10])

print('total training horse images:', len(os.listdir(train_horse_dir)))
print('total training human images:', len(os.listdir(train_human_dir)))
print('total validation horse images:', len(os.listdir(validation_horse_dir)))
print('total validation human images:', len(os.listdir(validation_human_dir)))


model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
    ])

model.summary()

from tensorflow.keras.optimizers import RMSprop
model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['acc'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1/255.0)
validation_dategen=ImageDataGenerator(rescale=1/255.0)

train_generator=train_datagen.flow_from_directory('D:/horse-or-human/train',
                                                  target_size=(300,300),
                                                  batch_size=128,
                                                  class_mode='binary')

validation_generator=validation_dategen.flow_from_directory('D:/horse-or-human/validation',
                                                            target_size=(300,300),
                                                            batch_size=32,
                                                            class_mode='binary')


history=model.fit_generator(train_generator,
                            steps_per_epoch=8,
                            epochs=2,
                            validation_data=validation_generator,
                            validation_steps=8,
                            verbose=2)


import numpy as np
from keras.preprocessing import image

path = r'D:/Skin_Detection/Caltech_faces/New_folder/image_0011.jpg'
img = image.load_img(path, target_size=(300, 300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])
if classes[0]>0.5:
  print('is a human')
else:
  print('is a horse')
