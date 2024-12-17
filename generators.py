import tensorflow as tf 
# from tensorflow import keras
# from tensorflow.keras import layers,Model

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