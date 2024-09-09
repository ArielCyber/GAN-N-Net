import os
import numpy as np
import tensorflow as tf

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


def get_samples_path(args):
    if args.run_name:
      samples_path = f'console_output/samples_{args.run_name}.npy'
    else:
      samples_path = f'console_output/samples_{os.path.basename(args.data_dir)}.npy'
    
    # os.makedirs(samples_path, exist_ok=True)
    return samples_path

def load_data(filepath):
    data = np.load(filepath)
    data = np.squeeze(data,axis=1)

    return data

def print_gpu_availability():
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
      raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))