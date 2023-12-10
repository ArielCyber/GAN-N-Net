import os
import tensorflow as tf 
import numpy as np 
from tensorflow import keras
import glob
from tensorflow.keras import layers,Model

def make_my_discriminator_model(imageDim): #(6_7)
    input = keras.Input(shape=(imageDim,imageDim,1))
    x = keras.layers.Dropout(0.3)(input)
    x = keras.layers.Conv2D(32,kernel_size=(5,5),strides = (1,1),padding = 'same')(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv2D(64,kernel_size =(3,3),strides =(1,1),padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    x = keras.layers.Conv2D(128,kernel_size=(2,2),strides = (1,1),padding='same')(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    #x = keras.layers.Dense(4)(x)
    model = Model(input,x,name='discriminator')
    return model

