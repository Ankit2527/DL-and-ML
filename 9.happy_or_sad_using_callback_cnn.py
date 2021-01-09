# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:59:07 2019

@author: Ankit
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

train_sad_dir=os.path.join('D:/happy-or-sad/sad')
train_happy_dir=os.path.join('D:/happy-or-sad/happy')

print('total sad training images', len(os.listdir(train_sad_dir)))
print('total happy training images', len(os.listdir(train_happy_dir)))


def train_sad_happy_model():
    
    class Callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs={}):
            if(logs.get('acc')>0.99):
                print('\nTraing reached 99%, hence stopping training')
                self.model.stop_training=True
                
    callbacks=Callback()

    model=tf.keras.Sequential([
        tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(1,activation='sigmoid')
        ])
                               
    model.summary()
                               
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['acc'])
    
    train_datagen=ImageDataGenerator(rescale=1/255.0)
    
    train_generator=train_datagen.flow_from_directory('D:/happy-or-sad',
                                                      target_size=(150,150),
                                                      batch_size=20,
                                                      class_mode='binary'
                                                      )
    

    history=model.fit_generator(train_generator,
                                steps_per_epoch=4,
                                epochs=15,
                                verbose=1,
                                callbacks=[callbacks])
    
    return history.epoch,history.history['acc']

train_sad_happy_model()
                                
                                