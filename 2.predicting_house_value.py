# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 11:29:07 2019

@author: Ankit
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

def house_model(y_new):
    model=keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    xs=np.array([1.0,2.0,3.0,4.0,7.0],dtype=float)
    ys=np.array([1.0,1.5,2.0,2.5,4.0],dtype=float)
    model.fit(xs,ys,epochs=1000)
    return model.predict(y_new)

prediction=house_model([10.0])
print(prediction)                   