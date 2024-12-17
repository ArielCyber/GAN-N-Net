
import argparse
import os
import tensorflow as tf 
import numpy as np 
from tensorflow import keras
from generate_plot import create_plot_comarison_graph
from tools import get_logger_path, get_model_path, get_samples_path, load_data, print_gpu_availability

class GanNNet(tf.keras.Model):
  def __init__(self,discriminator,generator, latent_dim, label_rate, batch_size):
    super(GanNNet,self).__init__()
    self.discriminator = discriminator
    self.generator = generator
    self.latent_dim = latent_dim
    self.label_rate = label_rate
    self.batch_size = batch_size
  def compile(self, d_optimizer, g_optimizer, loss_fn):
    super(GanNNet,self).compile()
    self.d_optimizer = d_optimizer
    self.g_optimizer = g_optimizer
    self.loss_fn = loss_fn
    
  def extended_labels(self, labels):
    extended_label = tf.concat([labels, tf.zeros([tf.shape(labels)[0], 1], tf.float32)], axis = 1)

    return extended_label

  def train_step(self,dataset):
    real_samples = dataset[0]
    labels = dataset[1]
    latent_vector = tf.random.normal(shape=(self.batch_size, self.latent_dim))
    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
      generated_samples = self.generator(latent_vector, training=True)
      real_pred = self.discriminator(real_samples, training=True)
      fake_pred = self.discriminator(generated_samples, training = True)
      labels = self.extended_labels(labels)
      d_loss, g_loss, train_acc, prec, rec = self.loss_fn(real_pred, fake_pred, labels, self.label_rate)
    d_grad = d_tape.gradient(d_loss,self.discriminator.trainable_variables)
    g_grad = g_tape.gradient(g_loss,self.generator.trainable_variables)
    self.d_optimizer.apply_gradients(zip(d_grad,self.discriminator.trainable_variables))
    self.g_optimizer.apply_gradients(zip(g_grad,self.generator.trainable_variables))

    return {"d_loss": d_loss, "g_loss": g_loss,"train_accuracy":train_acc,"precision":prec,"recall":rec}
  
  #evaluate step
  def test_step(self,dataset):
    features = dataset[0]
    labels = dataset[1]
    latent_vector = tf.random.normal(shape=(self.batch_size, self.latent_dim))
    
    generated_images = self.generator(latent_vector,training = False)
    real_features = self.discriminator(features,training=False)
    fake_features = self.discriminator(generated_images, training = False)
    labels = self.extended_labels(labels)
    d_loss,g_loss,acc,prec,rec = self.loss_fn(real_features,fake_features,labels,self.label_rate)
    return {"d_loss": d_loss, "g_loss": g_loss,"test_accuracy":acc,"test precision":prec,"test recall":rec}


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
    model = tf.keras.Model(input,x,name='discriminator')
    return model


def make_generator_model(inputSize, latent_dim, imageDim):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(inputSize*16, use_bias=False, input_shape=(latent_dim,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha = 1))
      
    model.add(tf.keras.layers.Reshape((imageDim,imageDim, 16)))
    assert model.output_shape == (None, imageDim,imageDim, 16) # Note: None is the batch size
    
    model.add(tf.keras.layers.Conv2DTranspose(8, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None,imageDim,imageDim, 8)  
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(4, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None,imageDim,imageDim, 4)    
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None,imageDim,imageDim, 1)
  
    return model