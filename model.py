# model.py
# by Umair Khan
# CS 410 - Spring 2020

# PyTorch class definition for SRCNN, implemented
# as specified in the original paper.

# Imports
import torch.nn as nn
import torch.nn.functional as F

# Class definition
class SRCNN(nn.Module):

    # Model setup.
    def __init__(self):

        # Initialize superclass
        super(SRCNN, self).__init__()

        # Define the three convolutional layers
        # (kernel sizes from paper, padding is to get dimensions correct)
        self.patch_ex = nn.Conv2d(1, 64, kernel_size = 9, padding = 4)
        self.nl_mapping = nn.Conv2d(64, 32, kernel_size = 1, padding = 0)
        self.reconstruction = nn.Conv2d(32, 1, kernel_size = 5, padding = 2)

    # Forward pass of input image.
    # Arguments:
    #  - x - tensor to push through network
    def forward(self, x):

        # First and second convolutional layers have ReLU
        y = F.relu(self.patch_ex(x))
        y = F.relu(self.nl_mapping(y))

        # Third layer does not have ReLU
        y = self.reconstruction(y)
        return y