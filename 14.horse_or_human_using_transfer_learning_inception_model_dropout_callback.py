# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 21:36:45 2019

@author: Ankit
"""

import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
import matplotlib.pyplot as plt

train_horse_dir=os.path.join('D:/Datasets/horse-or-human/train/horses')
train_human_dir=os.path.join('D:/Datasets/horse-or-human/train/humans')
validation_horse_dir=os.path.join('D:/Datasets/horse-or-human/validation/horses')
validation_human_dir=os.path.join('D:/Datasets/horse-or-human/validation/humans')

print(len(os.listdir(train_horse_dir)))
print(len(os.listdir(train_human_dir)))
print(len(os.listdir(validation_horse_dir)))
print(len(os.listdir(validation_human_dir)))

class callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('acc')>0.99):
            print('\n training stopped as reached 99%')
            self.model.stop_training=True
            
callbacks=callback()
            
local_weights_file='D:/Models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model=InceptionV3(input_shape=(150,150,3),
                              include_top=False,
                              weights=None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable=False
    
last_layer=pre_trained_model.get_layer('mixed7')
print('last layer output shape: ',last_layer.output_shape)
last_output=last_layer.output


x=layers.Flatten()(last_output)
x=layers.Dense(1024,activation='relu')(x)
x=layers.Dropout(0.2)(x)
x=layers.Dense(1,activation='sigmoid')(x)

model=Model(pre_trained_model.input,x)
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


train_generator=train_datagen.flow_from_directory('D:/Datasets/horse-or-human/train',
                                                  target_size=(150,150),
                                                  batch_size=20,
                                                  class_mode='binary')

validation_generator=validation_datagen.flow_from_directory('D:/Datasets/horse-or-human/validation',
                                                            target_size=(150,150),
                                                            batch_size=20,
                                                            class_mode='binary')

history=model.fit_generator(train_generator,
                    epochs=100,
                    validation_data=validation_generator,
                    verbose=2,
                    callbacks=[callbacks])

acc=history.history['acc']
val_acc=history.history['val_acc']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))

plt.plot(epochs,acc,'r',label='Training accuracy')
plt.plot(epochs,val_acc,'b',label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs,loss,'bo',label='Training_loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')     
plt.title('Traing and Validation loss')       
plt.legend()

plt.show()    