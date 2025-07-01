import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import glob
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from generate_plot import create_plot_comarison_graph
from tools import get_logger_path, get_model_path, get_samples_path, load_data, print_gpu_availability
from gan_n_net import GanNNet, make_my_discriminator_model, make_generator_model


class FlowDataset(Dataset):
    def __init__(self, data, labels):
        # Add channel dimension if not present
        if len(data.shape) == 3:  # (N, H, W) -> (N, 1, H, W)
            data = np.expand_dims(data, axis=1)
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def loss_values(d_real_features, fake_features, labels, label_rate, device, dense, batch_size):
    epsilon = 1e-8
    
    # Apply dense layer to get logits
    real_logits = dense(d_real_features)
    real_prob = F.softmax(real_logits, dim=-1)
    fake_logits = dense(fake_features)
    fake_prob = F.softmax(fake_logits, dim=-1)
    
    def d_loss_fn():
        # Cross entropy loss for real samples
        tmp = F.cross_entropy(real_logits, torch.argmax(labels, dim=1), reduction='none')
        
        # Use masking to determine the percentage of unlabeled samples to be used 
        labeled_mask = torch.zeros(batch_size, device=device)
        labeled_count = int(batch_size * label_rate)
        labeled_mask[:labeled_count] = 1.0
        
        D_L_supervised = torch.sum(labeled_mask * tmp) / torch.sum(labeled_mask)
        return D_L_supervised
    
    def g_loss_fn():
        # Feature matching
        real_moments = torch.mean(d_real_features, dim=0)
        generated_moments = torch.mean(fake_features, dim=0)
        G_L2 = torch.mean(torch.abs(real_moments - generated_moments))
        return G_L2
    
    # Calculate metrics
    predicted_labels = torch.argmax(real_prob, dim=1)
    true_labels = torch.argmax(labels, dim=1)
    
    # Convert to CPU for sklearn metrics
    predicted_cpu = predicted_labels.detach().cpu().numpy()
    true_cpu = true_labels.detach().cpu().numpy()
    
    train_accuracy = accuracy_score(true_cpu, predicted_cpu)
    precision = precision_score(true_cpu, predicted_cpu, average='weighted', zero_division=0)
    recall = recall_score(true_cpu, predicted_cpu, average='weighted', zero_division=0)
    
    d_loss = d_loss_fn()
    g_loss = g_loss_fn()
    
    return d_loss, g_loss, train_accuracy, precision, recall


def train_epoch(gan, dataloader, device, dense, epoch, csv_writer):
    total_d_loss = 0
    total_g_loss = 0
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    num_batches = 0
    
    for batch_idx, (real_samples, labels) in enumerate(dataloader):
        real_samples = real_samples.to(device)
        labels = labels.to(device)
        
        # Ensure batch size matches expected size
        if real_samples.size(0) != gan.batch_size:
            continue
            
        # Define loss function with current batch parameters
        def current_loss_fn(d_real_feat, fake_feat, ext_labels, label_rate):
            return loss_values(d_real_feat, fake_feat, ext_labels, label_rate, device, dense, gan.batch_size)
        
        # Update loss function
        gan.loss_fn = current_loss_fn
        
        # Train step
        metrics = gan.train_step(real_samples, labels)
        
        total_d_loss += metrics["d_loss"]
        total_g_loss += metrics["g_loss"]
        total_accuracy += metrics["train_accuracy"]
        total_precision += metrics["precision"]
        total_recall += metrics["recall"]
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}: '
                  f'D_Loss: {metrics["d_loss"]:.4f}, G_Loss: {metrics["g_loss"]:.4f}, '
                  f'Acc: {metrics["train_accuracy"]:.4f}')
    
    if num_batches > 0:
        avg_d_loss = total_d_loss / num_batches
        avg_g_loss = total_g_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_precision = total_precision / num_batches
        avg_recall = total_recall / num_batches
        
        # Log to CSV
        csv_writer.writerow([
            epoch, avg_d_loss, avg_g_loss, avg_accuracy, avg_precision, avg_recall
        ])
        
        print(f'Epoch {epoch} - Avg D_Loss: {avg_d_loss:.4f}, Avg G_Loss: {avg_g_loss:.4f}, '
              f'Avg Accuracy: {avg_accuracy:.4f}')
    
    return avg_d_loss, avg_g_loss, avg_accuracy


def evaluate_model(gan, dataloader, device, dense):
    gan.discriminator.eval()
    gan.generator.eval()
    
    total_d_loss = 0
    total_g_loss = 0
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    num_batches = 0
    
    with torch.no_grad():
        for real_samples, labels in dataloader:
            real_samples = real_samples.to(device)
            labels = labels.to(device)
            
            if real_samples.size(0) != gan.batch_size:
                continue
                
            def current_loss_fn(d_real_feat, fake_feat, ext_labels, label_rate):
                return loss_values(d_real_feat, fake_feat, ext_labels, label_rate, device, dense, gan.batch_size)
            
            gan.loss_fn = current_loss_fn
            metrics = gan.test_step(real_samples, labels)
            
            total_d_loss += metrics["d_loss"]
            total_g_loss += metrics["g_loss"]
            total_accuracy += metrics["test_accuracy"]
            total_precision += metrics["test_precision"]
            total_recall += metrics["test_recall"]
            num_batches += 1
    
    if num_batches > 0:
        avg_d_loss = total_d_loss / num_batches
        avg_g_loss = total_g_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_precision = total_precision / num_batches
        avg_recall = total_recall / num_batches
        
        print(f'Test - D_Loss: {avg_d_loss:.4f}, G_Loss: {avg_g_loss:.4f}, '
              f'Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}')
    
    return avg_d_loss, avg_g_loss, avg_accuracy


if __name__ == '__main__':
    device = print_gpu_availability()

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Directory containing the training data.')
    parser.add_argument('run_name', default='', help='Name of the run, will be used for the log file.')
    parser.add_argument('--test_split', type=float, default=0.3, required=False, help='Fraction of the data to use for the test set.')
    parser.add_argument('--val_split', type=float, default=0.1, required=False, help='Fraction of the data to use for the validation set.')
    parser.add_argument('--batch_size', type=int, default=64, required=False, help='Batch size to use for training.')
    parser.add_argument('--drop_reminder', default=True, action='store_true', required=False, help='Drop the last batch if it is not full.')
    parser.add_argument('--label_rate', type=float, default=1, required=False, help='Rate of labels in the training data. Default is 1 (all)')
    parser.add_argument('--epochs', type=int, default=200, required=False, help='Number of epochs to train. Default is 200.')
    parser.add_argument('--train_rate', type=float, default=1, required=False, help='Rate of training data. Default is 1 (all)')
    parser.add_argument('--save_model', default=False, required=False, help='Save the model after training.')

    args = parser.parse_args()
    
    # Fix data directory path 
    if args.data_dir[-1] != '/':
        args.data_dir += '/'
        
    print(args)    
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
        labelIndex += 1
        currentData = load_data(file)
        data.append(currentData)
        labels.append(np.full(currentData.shape[0], labelIndex))

    data = np.vstack(data)
    labels = np.hstack(labels)
    label_counts = pd.Series(labels).value_counts()
    
    # Calculate train, val, and test size
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

    # One-hot encode labels
    train_labels_onehot = np.eye(CLASS_NUM)[train_labels.astype(int)]
    val_labels_onehot = np.eye(CLASS_NUM)[val_labels.astype(int)]
    test_labels_onehot = np.eye(CLASS_NUM)[test_labels.astype(int)]

    # Create datasets and dataloaders
    train_dataset = FlowDataset(train_data, train_labels_onehot)
    val_dataset = FlowDataset(val_data, val_labels_onehot)
    test_dataset = FlowDataset(test_data, test_labels_onehot)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=args.drop_reminder)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=args.drop_reminder)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=args.drop_reminder)

    print("Train dataset length: ", len(train_dataloader))
    print("Validation dataset length: ", len(val_dataloader))
    print("Test dataset length: ", len(test_dataloader))

    # Generator
    g_model = make_generator_model(inputSize=input_size, latent_dim=latent_dim, imageDim=image_dim).to(device)
    print("Generator created")

    # Sanity check code:
    with torch.no_grad():
        genOut = g_model(torch.randn(BATCH_SIZE, latent_dim, device=device))
        print(f"Generator output shape: {genOut.shape}")

    # Discriminator
    d_model = make_my_discriminator_model(imageDim=image_dim).to(device)
    print("Discriminator created")

    # Sanity check code:
    with torch.no_grad():
        print(f"Discriminator output shape: {d_model(genOut).shape}")

    # Dense layer for classification
    dense = nn.Linear(128, CLASS_NUM+1).to(device)  # 128 is the output size from discriminator
    
    # Optimizers
    disc_optimizer = optim.Adam(d_model.parameters(), lr=1e-4)
    gen_optimizer = optim.Adam(g_model.parameters(), lr=1e-4)

    # Create GAN
    gan = GanNNet(discriminator=d_model, generator=g_model, latent_dim=latent_dim, 
                  label_rate=args.label_rate, batch_size=BATCH_SIZE, device=device)
    
    def dummy_loss_fn(d_real_feat, fake_feat, ext_labels, label_rate):
        return loss_values(d_real_feat, fake_feat, ext_labels, label_rate, device, dense, BATCH_SIZE)
    
    gan.compile(d_optimizer=disc_optimizer, g_optimizer=gen_optimizer, loss_fn=dummy_loss_fn)

    # CSV Logger
    logger_path = get_logger_path(args)
    csv_file = open(logger_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'd_loss', 'g_loss', 'accuracy', 'precision', 'recall'])

    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        train_epoch(gan, train_dataloader, device, dense, epoch, csv_writer)
        
        # Validation every 10 epochs
        if epoch % 10 == 0:
            print(f"Validation at epoch {epoch}:")
            evaluate_model(gan, val_dataloader, device, dense)

    csv_file.close()

    # Save model
    model_path = get_model_path(args)
    torch.save({
        'generator_state_dict': g_model.state_dict(),
        'discriminator_state_dict': d_model.state_dict(),
        'dense_state_dict': dense.state_dict(),
        'gen_optimizer_state_dict': gen_optimizer.state_dict(),
        'disc_optimizer_state_dict': disc_optimizer.state_dict(),
    }, model_path)
    print(f"Model saved to {model_path}")

    # Final evaluation
    print("Final evaluation on test set:")
    evaluate_model(gan, test_dataloader, device, dense)

    print(f'Train size: {train_size}')
    print(f'Test size: {test_size}')
    print(f'Train size after train rate: {train_size_using_rate}')
    
    # Generate samples
    print("Generating samples...")
    num_to_gen = BATCH_SIZE * 20
    samples = []
    predicted_labels = []
    
    gan.generator.eval()
    gan.discriminator.eval()
    
    with torch.no_grad():
        for i in range(0, num_to_gen, BATCH_SIZE):
            latent_vector = torch.randn(BATCH_SIZE, latent_dim, device=device)
            generated_samples = gan.generator(latent_vector)
            
            # Get predictions
            features = gan.discriminator(generated_samples)
            logits = dense(features)
            predictions = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            
            samples.append(generated_samples.cpu().numpy())
            predicted_labels.append(predictions.cpu().numpy())
    
    samples = np.vstack(samples)
    predicted_labels = np.hstack(predicted_labels)
    
    print(f'Samples shape: {samples.shape}')
    np.save(get_samples_path(args), samples)
    print("Generated label distribution:", np.bincount(predicted_labels))
    print("Original label distribution:")
    print(label_counts)
    print("The End")
