# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 00:12:44 2019

@author: Ankit
"""

import tensorflow as tf
from tensorflow import keras

mnist=tf.keras.datasets.mnist
(training_images,training_labels),(test_images,test_labels)=mnist.load_data()

training_images=training_images.reshape(training_images.shape[0],training_images.shape[1],training_images.shape[2],1)
training_images=training_images/255.0

test_images=test_images.reshape(test_images.shape[0],test_images.shape[1],test_images.shape[2],1)
test_images=test_images/255.0

def train_mnist_conv():
    
    class Callbacks(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs={}):
            if(logs.get('acc')>0.998):
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training=True
                
    callbacks=Callbacks()
    
    model=tf.keras.Sequential([
        tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(10,activation='softmax')
        ])
    
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    history=model.fit(training_images,training_labels,epochs=20,callbacks=[callbacks])
    
    return history.epoch,history.history['acc']

_,_=train_mnist_conv()

