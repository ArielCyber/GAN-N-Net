
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from generate_plot import create_plot_comarison_graph
from tools import get_logger_path, get_model_path, get_samples_path, load_data, print_gpu_availability

class GanNNet(nn.Module):
    def __init__(self, discriminator, generator, latent_dim, label_rate, batch_size, device):
        super(GanNNet, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.label_rate = label_rate
        self.batch_size = batch_size
        self.device = device
        
    def compile(self, d_optimizer, g_optimizer, loss_fn):
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        
    def extended_labels(self, labels):
        batch_size = labels.shape[0]
        zeros = torch.zeros(batch_size, 1, device=self.device)
        extended_label = torch.cat([labels, zeros], dim=1)
        return extended_label
    
    def train_step(self, real_samples, labels):
        # Set models to training mode
        self.discriminator.train()
        self.generator.train()
        
        # Generate random latent vectors
        latent_vector = torch.randn(self.batch_size, self.latent_dim, device=self.device)
        
        # Generate fake samples
        generated_samples = self.generator(latent_vector)
        
        # Get discriminator features
        real_pred = self.discriminator(real_samples)
        fake_pred = self.discriminator(generated_samples.detach())  # Detach for discriminator training
        
        # Extend labels for discriminator classification
        extended_labels = self.extended_labels(labels)
        
        # Compute losses
        d_loss, g_loss, train_acc, prec, rec = self.loss_fn(real_pred, fake_pred, extended_labels, self.label_rate)
        
        # Update discriminator
        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()
        
        # Update generator (need fresh fake samples for generator training)
        fake_pred_for_g = self.discriminator(generated_samples)
        _, g_loss, _, _, _ = self.loss_fn(real_pred.detach(), fake_pred_for_g, extended_labels, self.label_rate)
        
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item(),
            "train_accuracy": train_acc,
            "precision": prec,
            "recall": rec
        }
    
    def test_step(self, features, labels):
        # Set models to evaluation mode
        self.discriminator.eval()
        self.generator.eval()
        
        with torch.no_grad():
            latent_vector = torch.randn(self.batch_size, self.latent_dim, device=self.device)
            
            generated_images = self.generator(latent_vector)
            real_features = self.discriminator(features)
            fake_features = self.discriminator(generated_images)
            extended_labels = self.extended_labels(labels)
            d_loss, g_loss, acc, prec, rec = self.loss_fn(real_features, fake_features, extended_labels, self.label_rate)
            
        return {
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item(),
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec
        }


class Discriminator(nn.Module):
    def __init__(self, image_dim):
        super(Discriminator, self).__init__()
        self.dropout1 = nn.Dropout(0.3)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.leaky_relu1 = nn.LeakyReLU(inplace=False)
        self.dropout2 = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=False)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        x = self.dropout1(x)
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.dropout2(x)
        x = self.conv2(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu2(x)
        x = self.conv3(x)
        x = self.leaky_relu3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x


def make_my_discriminator_model(imageDim):
    return Discriminator(imageDim)


class Generator(nn.Module):
    def __init__(self, input_size, latent_dim, image_dim):
        super(Generator, self).__init__()
        self.image_dim = image_dim
        
        self.dense = nn.Linear(latent_dim, input_size * 16, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(input_size * 16)
        self.leaky_relu1 = nn.LeakyReLU(1.0)
        
        self.conv_transpose1 = nn.ConvTranspose2d(16, 8, kernel_size=5, stride=1, padding=2, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(8)
        self.leaky_relu2 = nn.LeakyReLU()
        
        self.conv_transpose2 = nn.ConvTranspose2d(8, 4, kernel_size=5, stride=1, padding=2, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(4)
        self.leaky_relu3 = nn.LeakyReLU()
        
        self.conv_transpose3 = nn.ConvTranspose2d(4, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.dense(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu1(x)
        
        # Reshape to (batch_size, 16, image_dim, image_dim)
        x = x.view(x.size(0), 16, self.image_dim, self.image_dim)
        
        x = self.conv_transpose1(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu2(x)
        
        x = self.conv_transpose2(x)
        x = self.batch_norm3(x)
        x = self.leaky_relu3(x)
        
        x = self.conv_transpose3(x)
        x = self.tanh(x)
        
        return x


def make_generator_model(inputSize, latent_dim, imageDim):
    return Generator(inputSize, latent_dim, imageDim)