# GAN-N-Net Project README

## Overview

The GAN-N-Net repo contains code written as part of research to provide a method for reproducing the results presented in the associated paper. The codebase includes implementation of models with and without the proposed GAN-N-Net model enhancements.

## Repository Structure

- `mini.py`: Script to obtain results using our GAN-N-Net model.
- `mini100.py`: Script to obtain results without using our GAN-N-Net model.

## Getting Started

To reproduce the paper results, follow these steps:

### Prerequisites

Ensure you have the requairements package installed
you can use requirements.txt file
pip install -r requirements.txt

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/[username]/GAN-N-Net.git
cd GAN-N-Net

pip install -r requirements.txt

# To get results with our model

python mini.py

# To get results without our model
python mini100.py