# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 10:44:05 2020

@author: Ankit
"""

import os
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import matplotlib.pyplot as plt

train_paper_dir=os.path.join('D:/Datasets/Rock_papr_scissor/rps/paper')
train_rock_dir=os.path.join('D:/Datasets/Rock_papr_scissor/rps/rock')
train_scissor_dir=os.path.join('D:/Datasets/Rock_papr_scissor/rps/scissors')

validation_paper_dir=os.path.join('D:/Datasets/Rock_papr_scissor/rps-test-set/paper')
validation_rock_dir=os.path.join('D:/Datasets/Rock_papr_scissor/rps-test-set/rock')
validation_scissor_dir=os.path.join('D:/Datasets/Rock_papr_scissor/rps-test-set/scissors')

print(len(os.listdir(train_paper_dir)))
print(len(os.listdir(train_rock_dir)))
print(len(os.listdir(train_scissor_dir)))
print(len(os.listdir(validation_paper_dir)))
print(len(os.listdir(validation_rock_dir)))
print(len(os.listdir(validation_scissor_dir)))


model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(3,activation='softmax')
    ])

model.summary()

model.compile(loss='categorical_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['acc'])

train_datagen=ImageDataGenerator(rescale=1./255,                                                 
                                 rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')

validation_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory('D:/Datasets/Rock_papr_scissor/rps',
                                                  target_size=(150,150),
                                                  class_mode='categorical')

validation_generator=validation_datagen.flow_from_directory('D:/Datasets/Rock_papr_scissor/rps-test-set',
                                                              target_size=(150,150),
                                                              class_mode='categorical')

history=model.fit_generator(train_generator,
                            epochs=25,
                            validation_data=validation_generator,
                            verbose=1)

model.save("rps.h5")

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))

plt.plot(epochs,acc,'r',label='Training accuracy')
plt.plot(epochs,val_acc,'b',label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs,loss,'r',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation_loss')
plt.legend()