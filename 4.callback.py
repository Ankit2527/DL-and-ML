# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 12:40:16 2019

@author: Ankit
"""

import tensorflow as tf
from tensorflow import keras

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if (logs.get('loss')<0.4):
            print('\nReached 60% accuracy, so cancelling')
            self.model.stop_training=True
            
callbacks=myCallback()


mnist=tf.keras.datasets.fashion_mnist
(training_images,training_labels),(test_images,test_labels)=mnist.load_data()

training_images=training_images/255.0
test_images=test_images/255.0

model=tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
    ])

model.compile(quantizer='adam',loss='sparse_categorical_crossentropy')
model.fit(training_images,training_labels,epochs=10,callbacks=[callbacks])