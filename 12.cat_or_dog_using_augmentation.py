# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 10:27:37 2019

@author: Ankit
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

train_cat_dir=os.path.join('D:/Datasets/cats_and_dogs_filtered/train/cats')
train_dog_dir=os.path.join('D:/Datasets/cats_and_dogs_filtered/train/dogs')
validation_cat_dir=os.path.join('D:/Datasets/cats_and_dogs_filtered/validation/cats')
validation_dog_dir=os.path.join('D:/Datasets/cats_and_dogs_filtered/validation/dogs')

print(len(os.listdir(train_cat_dir)))
print(len(os.listdir(train_dog_dir)))
print(len(os.listdir(validation_cat_dir)))
print(len(os.listdir(validation_dog_dir)))


model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
    ])

model.summary()

model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=1e-4),metrics=['acc'])

train_datagen=ImageDataGenerator(rescale=1./255,
                                 rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')

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
                    epochs=100,
                    validation_data=validation_generator,
                    validation_steps=50,
                    verbose=2)

acc=history.history['acc']
val_acc=history.history['val_acc']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))

plt.plot(epochs,acc,'bo',label='Training accuracy')
plt.plot(epochs,val_acc,'b',label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs,loss,'bo',label='Training_loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')     
plt.title('Traing and Validation loss')       
plt.legend()

plt.show()                                      
