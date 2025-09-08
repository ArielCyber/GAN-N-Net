### GAN-N-Net Generator Architecture

| Layer            | Details                                                       | Activation Function        |
|------------------|---------------------------------------------------------------|----------------------------|
| Fully Connected  | Output size = 32×32×16, BatchNorm                             | LeakyReLU (α = 1.0)        |
| Reshape          | Output shape: (32, 32, 16)                                   | –                          |
| Conv2D           | 8 filters, 5×5 kernel, stride 1, same padding, BatchNorm      | LeakyReLU (α = 0.3)        |
| Conv2D           | 4 filters, 5×5 kernel, stride 1, same padding, BatchNorm      | LeakyReLU (α = 0.3)        |
| Conv2D           | 1 filter, 5×5 kernel, stride 1, same padding, BatchNorm       | tanh                       |
