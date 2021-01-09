# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 13:59:36 2019

@author: Ankit
"""

import tensorflow as tf
from tensorflow import keras

mnist=tf.keras.datasets.mnist
(training_images,training_labels),(test_images,test_labels)=mnist.load_data()

training_images=training_images/255.0
test_images=test_images/255.0

def train_mnist():
    
    class Callbacks(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs={}):
            if(logs.get('acc')>0.99):
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training=True
                
    callbacks=Callbacks()
    
    model=tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024,activation=tf.nn.relu),
        tf.keras.layers.Dense(10,activation=tf.nn.softmax)
        ])
    
    model.compile(quantizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
    history=model.fit(training_images,training_labels,epochs=10,callbacks=[callbacks])
    
    return history.epoch,history.history['acc']

train_mnist()
    
    