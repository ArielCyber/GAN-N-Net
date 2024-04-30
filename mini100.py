import argparse
import os
import tensorflow as tf 
import numpy as np 
from tensorflow import keras
import glob
from generate_plot import create_plot_comarison_graph
from tensorflow.keras import layers, Model
from discriminators import make_my_discriminator_model


def load_data(filepath):
    data = np.load(filepath)
    data = np.squeeze(data,axis=1)

    return data


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


#discriminator loss
def loss_values(d_real_features,fake_features,labels,label_rate):
    
    epsilon = 1e-8
    real_logits = dense(d_real_features)
    real_prob = tf.nn.softmax(real_logits)
    fake_logits =dense(fake_features)
    fake_prob = tf.nn.softmax(fake_logits)
    def d_loss_fn():
      
      tmp = tf.nn.softmax_cross_entropy_with_logits(logits = real_logits,
                                                  labels = labels)
      # use masking to determine  the percentage of unlabeled samples to be used 
      labeled_mask = np.zeros([BATCH_SIZE], dtype = np.float32)
      labeled_count = np.int_(BATCH_SIZE * label_rate)
      labeled_mask[range(labeled_count)] = 1.0
      D_L_supervised = tf.reduce_sum(labeled_mask * tmp) / tf.reduce_sum(labeled_mask)

      disc_loss = D_L_supervised
      return disc_loss

    def g_loss_fn():
     # Feature matching
     real_moments = tf.reduce_mean(d_real_features, axis = 0)
     generated_moments = tf.reduce_mean(fake_features, axis = 0)
     G_L2 = tf.reduce_mean(tf.abs(real_moments - generated_moments))
     #gen_loss = G_L1 +G_L2
     return G_L2
  
    train_accuracy.update_state(labels,real_prob )
    precision.update_state(labels,real_prob)
    recall.update_state(labels,real_prob)
    d_loss = d_loss_fn()
    g_loss = g_loss_fn()
    return d_loss,g_loss,train_accuracy.result(),precision.result(),recall.result()


class GanNNet(Model):
  def __init__(self,discriminator,generator, latent_dim, label_rate):
    super(GanNNet,self).__init__()
    self.discriminator = discriminator
    self.generator = generator
    self.latent_dim = latent_dim
    self.label_rate = label_rate
  def compile(self,d_optimizer,g_optimizer,loss_fn):
    super(GanNNet,self).compile()
    self.d_optimizer = d_optimizer
    self.g_optimizer = g_optimizer
    self.loss_fn = loss_fn
    
  def extended_labels(self,labels):
    extended_label = tf.concat([labels, tf.zeros([tf.shape(labels)[0], 1],tf.float32)], axis = 1)

    return extended_label

  def train_step(self,dataset):
    real_samples = dataset[0]
    labels = dataset[1]
    latent_vector = tf.random.normal(shape =(int(BATCH_SIZE*(1-self.label_rate)), self.latent_dim))
    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
      generated_samples = self.generator(latent_vector,training=True)
      real_pred = self.discriminator(real_samples, training=True)
      fake_pred = self.discriminator(generated_samples, training = True)
      labels = self.extended_labels(labels)
      d_loss, g_loss, train_acc, prec, rec = self.loss_fn(real_pred, fake_pred, labels,self.label_rate)
    d_grad = d_tape.gradient(d_loss,self.discriminator.trainable_variables)
    g_grad = g_tape.gradient(g_loss,self.generator.trainable_variables)
    self.d_optimizer.apply_gradients(zip(d_grad,self.discriminator.trainable_variables))
    self.g_optimizer.apply_gradients(zip(g_grad,self.generator.trainable_variables))

    return {"d_loss": d_loss, "g_loss": g_loss,"train_accuracy":train_acc,"precision":prec,"recall":rec}
  
  #evaluate step
  def test_step(self,dataset):
    features = dataset[0]
    labels = dataset[1]
    latent_vector = tf.random.normal(shape=(BATCH_SIZE, self.latent_dim))
    
    generated_images = self.generator(latent_vector,training = False)
    real_features = self.discriminator(features,training=False)
    fake_features = self.discriminator(generated_images, training = False)
    labels = self.extended_labels(labels)
    d_loss,g_loss,acc,prec,rec = self.loss_fn(real_features,fake_features,labels,self.label_rate)
    return {"d_loss": d_loss, "g_loss": g_loss,"test_accuracy":acc,"test precision":prec,"test recall":rec}


def get_logger_path(args):
    os.makedirs('console_output', exist_ok=True)
    if args.run_name:
      logger_path = f'console_output/training_{args.run_name}.csv'
    else:
      logger_path = f'console_output/training_{os.path.basename(args.data_dir)}.csv'
    return logger_path

def get_model_path(args):
    if args.run_name:
      model_path = f'console_output/model_{args.run_name}/model.h5'
    else:
      model_path = f'console_output/model_{os.path.basename(args.data_dir)}/model.h5'
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    return model_path

if __name__ == '__main__':
  # check if gpu is available
  device_name = tf.test.gpu_device_name()
  if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
  print('Found GPU at: {}'.format(device_name))

  parser = argparse.ArgumentParser()
  parser.add_argument('data_dir', help='Directory containing the training data.')
  parser.add_argument('run_name', default='', help='Name of the run, will be used for the log file.')
  parser.add_argument('--test_split', type=float, default=0.1, required=False, help='Fraction of the data to use for the test set.')
  parser.add_argument('--val_split', type=float, default=0.3, required=False, help='Fraction of the data to use for the validation set.')
  parser.add_argument('--batch_size', type=int, default=64, required=False, help='Batch size to use for training.')
  parser.add_argument('--drop_reminder', default=True, action='store_true', required=False, help='Drop the last batch if it is not full. Default True, action is store_true which means that if this argument is not specified it will be treated as True.')
  parser.add_argument('--label_rate', type=float, default=1, required=False, help='Rate of labels in the training data. Default is 1 (all)')
  parser.add_argument('--epochs', type=int, default=200, required=False, help='Number of epochs to train. Default is 200.')
  parser.add_argument('--train_rate', type=float, default=1, required=False, help='Rate of training data. Default is 1 (all)')
  parser.add_argument('--save_model', default=False, required=False, help='Save the model after training.')


  args = parser.parse_args()
  # fix data directory path 
  if args.data_dir[-1] != '/':
      args.data_dir += '/'
      
  files = glob.glob(args.data_dir + "*")
  CLASS_NUM = len(files)
  BATCH_SIZE = args.batch_size
  
  # GAN random input vector size
  latent_dim = 32*32*3
  data = []
  labels = []
  labelIndex = -1
  # Load the PIMs
  for file in files:
    print(f"working on: {file}")
    if file.endswith('.npy'):
      labelIndex += 1
      dataForFile = load_data(file)
      data.append(dataForFile)
      labelsForFile = np.ones(dataForFile.__len__()) * labelIndex
      labels.append(labelsForFile)

  data = np.vstack(data)
  labels = np.hstack(labels)
  data_len = data.__len__()
  test_size = int(args.test_split * data_len)
  val_size = int(args.val_split * args.train_rate * data_len)
  train_size = data_len - test_size
  train_size_using_rate = int(args.train_rate * train_size)


  image_dim = data.shape[-1]
  input_size = image_dim**2

  test_inds = np.random.choice(range(data_len), size=test_size, replace=False)

  test_data = data[test_inds]
  test_labels = labels[test_inds]

  train_data = np.delete(data, test_inds, 0)
  train_labels = np.delete(labels, test_inds)

  train_data = tf.cast(train_data, tf.float32)
  test_data = tf.cast(test_data, tf.float32)

  train_labels = tf.cast(train_labels, tf.int32)
  train_labels = tf.one_hot(train_labels, CLASS_NUM)

  test_labels = tf.cast(test_labels, tf.int32)
  test_labels = tf.one_hot(test_labels, CLASS_NUM)


  train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
  train_dataset = train_dataset.shuffle(data_len)
  print("full train_dataset length: ", train_dataset.__len__())
  train_dataset = train_dataset.take(train_size_using_rate)
  print("train_dataset after take only train_rate length: ", train_dataset.__len__())

  train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=args.drop_reminder)

  test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
  test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=args.drop_reminder)

  print("test_dataset length: ", test_dataset.__len__())
  print("train_dataset.take(1): ", train_dataset.take(1))

  # Generator
  g_model = make_generator_model(inputSize=input_size, latent_dim=latent_dim, imageDim=image_dim)
  print(g_model.summary())

  # sanity check code:
  genOut = g_model(tf.random.normal(shape =(BATCH_SIZE, latent_dim)))
  print(genOut.shape)

  # Discriminator
  d_model = make_my_discriminator_model(imageDim=image_dim)


  print(d_model.summary())

  # sanity check code:
  print(d_model(genOut).shape)


  dense = keras.layers.Dense(CLASS_NUM+1)
  train_accuracy = tf.keras.metrics.CategoricalAccuracy(name = 'train_accuracy')
  precision = tf.keras.metrics.Precision()
  recall = tf.keras.metrics.Recall()

  disc_optimizer = keras.optimizers.Adam(1e-4)
  gen_optimizer = keras.optimizers.Adam(1e-4)


  gan = GanNNet(discriminator=d_model,generator=g_model,latent_dim=latent_dim, label_rate=args.label_rate)
  gan.compile(d_optimizer=disc_optimizer, g_optimizer= gen_optimizer,loss_fn=loss_values)

  csv_logger = keras.callbacks.CSVLogger(get_logger_path(args))

  cbks = [csv_logger]

  history = gan.fit(train_dataset, epochs=args.epochs, validation_data=train_dataset.take(val_size), callbacks=cbks)

  gan.save_weights(get_model_path(args))

  gan.evaluate(test_dataset.take(test_size))


  print(f'Train size: {train_size}')
  print(f'Test size: {test_size}')
  print(f'Train size after train rate: {train_size_using_rate}')
  print("The End")