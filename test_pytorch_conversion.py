#!/usr/bin/env python3
"""
Test script to verify PyTorch conversion works correctly
"""

import torch
import torch.nn as nn
import numpy as np
from gan_n_net import GanNNet, make_my_discriminator_model, make_generator_model
from tools import print_gpu_availability

def test_models():
    """Test that the PyTorch models can be instantiated and run forward pass"""
    
    device = print_gpu_availability()
    
    # Test parameters
    batch_size = 4
    image_dim = 32
    latent_dim = 32*32*3
    input_size = image_dim**2
    num_classes = 9
    
    print("Testing Generator...")
    generator = make_generator_model(input_size, latent_dim, image_dim).to(device)
    
    # Test generator forward pass
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_images = generator(z)
    print(f"Generator output shape: {fake_images.shape}")
    assert fake_images.shape == (batch_size, 1, image_dim, image_dim), f"Expected {(batch_size, 1, image_dim, image_dim)}, got {fake_images.shape}"
    
    print("Testing Discriminator...")
    discriminator = make_my_discriminator_model(image_dim).to(device)
    
    # Test discriminator forward pass
    features = discriminator(fake_images)
    print(f"Discriminator output shape: {features.shape}")
    assert features.shape == (batch_size, 128), f"Expected {(batch_size, 128)}, got {features.shape}"
    
    print("Testing GAN...")
    gan = GanNNet(discriminator, generator, latent_dim, label_rate=1.0, batch_size=batch_size, device=device)
    
    # Test extended labels
    labels = torch.eye(num_classes, device=device)[:batch_size]  # One-hot encoded
    extended_labels = gan.extended_labels(labels)
    print(f"Extended labels shape: {extended_labels.shape}")
    assert extended_labels.shape == (batch_size, num_classes + 1), f"Expected {(batch_size, num_classes + 1)}, got {extended_labels.shape}"
    
    print("Testing data flow...")
    # Create fake data
    real_samples = torch.randn(batch_size, 1, image_dim, image_dim, device=device)
    
    # Test that we can run inference
    with torch.no_grad():
        real_pred = discriminator(real_samples)
        fake_pred = discriminator(fake_images)
        print(f"Real prediction shape: {real_pred.shape}")
        print(f"Fake prediction shape: {fake_pred.shape}")
    
    print("✅ All tests passed! PyTorch conversion is working correctly.")
    
    return True

def test_training_components():
    """Test training-related components"""
    
    device = print_gpu_availability()
    
    batch_size = 4
    image_dim = 32
    latent_dim = 32*32*3
    input_size = image_dim**2
    num_classes = 9
    
    # Create models
    generator = make_generator_model(input_size, latent_dim, image_dim).to(device)
    discriminator = make_my_discriminator_model(image_dim).to(device)
    dense = nn.Linear(128, num_classes + 1).to(device)
    
    # Create optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    
    # Test that gradients can be computed
    real_samples = torch.randn(batch_size, 1, image_dim, image_dim, device=device)
    labels = torch.eye(num_classes, device=device)[:batch_size]
    
    # Test discriminator training
    real_features = discriminator(real_samples)
    real_logits = dense(real_features)
    real_loss = nn.CrossEntropyLoss()(real_logits, torch.argmax(labels, dim=1))
    
    d_optimizer.zero_grad()
    real_loss.backward()
    d_optimizer.step()
    
    # Test generator training (separate forward pass to avoid gradient conflicts)
    latent_vector = torch.randn(batch_size, latent_dim, device=device)
    fake_samples = generator(latent_vector)
    
    # Fresh forward pass for generator training
    with torch.no_grad():
        real_features_target = discriminator(real_samples)
    
    fake_features = discriminator(fake_samples)
    fake_loss = nn.MSELoss()(torch.mean(real_features_target, dim=0), torch.mean(fake_features, dim=0))
    
    g_optimizer.zero_grad()
    fake_loss.backward()
    g_optimizer.step()
    
    print("✅ Training components test passed!")
    
    return True

if __name__ == "__main__":
    print("Testing PyTorch GAN implementation...")
    print("=" * 50)
    
    try:
        test_models()
        print()
        test_training_components()
        print()
        print("🎉 All tests completed successfully!")
        print("The PyTorch conversion is ready to use.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
