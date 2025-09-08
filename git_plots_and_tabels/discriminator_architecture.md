### GAN-N-Net Discriminator Architecture

| Layer                | Details                                                     | Activation Function        |
|-----------------------|-------------------------------------------------------------|----------------------------|
| Input                | Image sample of shape (32, 32, 1)                           | –                          |
| Dropout              | Dropout layer with rate = 0.3                               | –                          |
| Conv2D               | 32 filters, 5×5 kernel, stride 1, same padding              | LeakyReLU (α = 0.3)        |
| Dropout              | Dropout layer with rate = 0.5                               | –                          |
| Conv2D               | 64 filters, 3×3 kernel, stride 1, same padding, BatchNorm   | LeakyReLU (α = 0.2)        |
| Conv2D               | 128 filters, 2×2 kernel, stride 1, same padding             | LeakyReLU (α = 0.2)        |
| Global Avg Pooling   | Global Average Pooling 2D layer                             | –                          |
