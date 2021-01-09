# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 23:07:23 2019

@author: Ankit
"""

import tensorflow as tf 
from tensorflow import keras

mnist=tf.keras.datasets.fashion_mnist
(training_images,training_labels),(test_images,test_labels)=mnist.load_data()

training_images=training_images.reshape(60000,28,28,1)
test_images=test_images.reshape(10000,28,28,1)
training_images=training_images/255.0
test_images=test_images/255.0

model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
    ])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
model.summary()
model.fit(training_images,training_labels,epochs=10)
test_loss=model.evaluate(test_images,test_labels)