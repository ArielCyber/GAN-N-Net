import argparse
import os
import tensorflow as tf 
import numpy as np 
import glob
from generate_plot import create_plot_comarison_graph
from tools import get_logger_path, get_model_path, get_samples_path, load_data, print_gpu_availability
from gan_n_net import GanNNet, make_my_discriminator_model, make_generator_model
from sklearn.model_selection import train_test_split
import pandas as pd

def loss_values(d_real_features,fake_features,labels,label_rate):
    
    epsilon = 1e-8
    real_logits = dense(d_real_features)
    real_prob = tf.nn.softmax(real_logits)
    fake_logits = dense(fake_features)
    fake_prob = tf.nn.softmax(fake_logits)
    def d_loss_fn():
      
      tmp = tf.nn.softmax_cross_entropy_with_logits(logits = real_logits,
                                                  labels = labels)
      # use masking to determine  the percentage of unlabeled samples to be used 
      labeled_mask = np.zeros([BATCH_SIZE], dtype = np.float32)
      labeled_count = np.int_(BATCH_SIZE * label_rate)
      labeled_mask[range(labeled_count)] = 1.0
      D_L_supervised = tf.reduce_sum(labeled_mask * tmp) / tf.reduce_sum(labeled_mask)

      prob_fake_be_fake = fake_prob[:, -1] + epsilon
      tmp_log = tf.math.log(prob_fake_be_fake)
      D_L_unsupervised2 = -1 * tf.reduce_mean(tmp_log)
      

      disc_loss = D_L_supervised + D_L_unsupervised2


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




if __name__ == '__main__':
  print_gpu_availability()

  parser = argparse.ArgumentParser()
  parser.add_argument('data_dir', help='Directory containing the training data.')
  parser.add_argument('run_name', default='', help='Name of the run, will be used for the log file.')
  parser.add_argument('--test_split', type=float, default=0.3, required=False, help='Fraction of the data to use for the test set.')
  parser.add_argument('--val_split', type=float, default=0.1, required=False, help='Fraction of the data to use for the validation set.')
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
      labelsForFile = np.ones(dataForFile.shape[0]) * labelIndex
      labels.append(labelsForFile)

  data = np.vstack(data)
  labels = np.hstack(labels)
  label_counts = pd.Series(labels).value_counts()

  # calculate train, val, and test size
  data_len = data.shape[0]
  test_size = int(args.test_split * data_len)
  val_size = int(args.val_split * args.train_rate * data_len)
  train_size = data_len - test_size
  train_size_using_rate = int(args.train_rate * train_size)


  image_dim = data.shape[-1]
  input_size = image_dim**2

  # Split data into test and train/validation sets
  train_val_data, test_data, train_val_labels, test_labels = train_test_split(
    data, labels, test_size=test_size, stratify=labels, random_state=42
  )

  # Further split train/validation into train and validation sets
  train_data, val_data, train_labels, val_labels = train_test_split(
    train_val_data, train_val_labels, test_size=val_size, stratify=train_val_labels, random_state=42
  )

  # Cast and one-hot encode labels
  train_data = tf.cast(train_data, tf.float32)
  val_data = tf.cast(val_data, tf.float32)
  test_data = tf.cast(test_data, tf.float32)

  train_labels = tf.cast(train_labels, tf.int32)
  train_labels = tf.one_hot(train_labels, CLASS_NUM)

  val_labels = tf.cast(val_labels, tf.int32)
  val_labels = tf.one_hot(val_labels, CLASS_NUM)
  
  test_labels = tf.cast(test_labels, tf.int32)
  test_labels = tf.one_hot(test_labels, CLASS_NUM)

  train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
  val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
  test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))

  # Shuffle and batch datasets
  train_dataset = train_dataset.shuffle(train_size).batch(BATCH_SIZE, drop_remainder=args.drop_reminder)
  val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=args.drop_reminder)
  test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=args.drop_reminder)

  print("Train dataset length: ", len(list(train_dataset)))
  print("Validation dataset length: ", len(list(val_dataset)))
  print("Test dataset length: ", len(list(test_dataset)))
  print("Sample train batch: ", list(train_dataset.take(1)))

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


  dense = tf.keras.layers.Dense(CLASS_NUM+1)
  train_accuracy = tf.keras.metrics.CategoricalAccuracy(name = 'train_accuracy')
  precision = tf.keras.metrics.Precision()
  recall = tf.keras.metrics.Recall()

  disc_optimizer = tf.keras.optimizers.Adam(1e-4)
  gen_optimizer = tf.keras.optimizers.Adam(1e-4)


  gan = GanNNet(discriminator=d_model, generator=g_model, latent_dim=latent_dim, label_rate=args.label_rate, batch_size=BATCH_SIZE)
  gan.compile(d_optimizer=disc_optimizer, g_optimizer=gen_optimizer, loss_fn=loss_values)

  csv_logger = tf.keras.callbacks.CSVLogger(get_logger_path(args))

  cbks = [csv_logger]

  history = gan.fit(train_dataset, epochs=args.epochs, validation_data=train_dataset.take(val_size), callbacks=cbks)

  gan.save(get_model_path(args))

  gan.evaluate(test_dataset.take(test_size))


  print(f'Train size: {train_size}')
  print(f'Test size: {test_size}')
  print(f'Train size after train rate: {train_size_using_rate}')
  
# generate samples
  num_to_gen = BATCH_SIZE * 20
  samples = np.empty((num_to_gen, image_dim, image_dim, 1))
  predicted_labels = np.empty(num_to_gen, dtype=np.int32)
  for i in range(0, num_to_gen, BATCH_SIZE):
    generated_images = gan.generator(tf.random.normal(shape=(BATCH_SIZE, latent_dim)), training=False)
    samples[i:i+BATCH_SIZE] = generated_images.numpy()
    predicted_labels[i:i+BATCH_SIZE] = np.argmax(gan.discriminator(generated_images).numpy(), axis=1)  
  print(f'samples shape:{samples.shape}')
  np.save(get_samples_path(args), samples)
  print(np.bincount(predicted_labels))
  print("distribution of labels:")
  print(label_counts)
  print("The End")