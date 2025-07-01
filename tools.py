import os
import numpy as np
import torch

def get_logger_path(args):
    os.makedirs('console_output', exist_ok=True)
    if args.run_name:
      logger_path = f'console_output/training_{args.run_name}.csv'
    else:
      logger_path = f'console_output/training_{os.path.basename(args.data_dir)}.csv'
    return logger_path

def get_model_path(args):
    if args.run_name:
      model_path = f'console_output/model_{args.run_name}/model.pth'
    else:
      model_path = f'console_output/model_{os.path.basename(args.data_dir)}/model.pth'
    
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
  if len(data.shape) > 3:
    data = np.squeeze(data,axis=1)

  return data

def print_gpu_availability():
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        print(f'Found GPU: {device}')
        return torch.device('cuda')
    else:
        print('GPU not available, using CPU')
        return torch.device('cpu')