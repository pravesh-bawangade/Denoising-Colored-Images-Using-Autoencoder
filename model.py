"""
Author: Pravesh Bawangade
File Name: main.py

"""

import torch.nn as nn
import torch.nn.functional as F


# define the NN architecture
class ConvDenoiser(nn.Module):
    """
    Network Architecture
    """
    def __init__(self):
        """
        Defining Three CNN layers, A max Pooling layer, Three Transpose CNN layers and a output cnn layer.
        """
        super(ConvDenoiser, self).__init__()

        # encoder layers #

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)

        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)

        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)

        # self.conv4 = nn.Conv2d(16, 8, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        ## decoder layers ##
        # transpose layer, a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(8, 8, 2, stride=2)  # kernel_size=3 to get to a 7x7 image output
        # two more transpose layers with a kernel of 2
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 32, 2, stride=2)

        # one, final, normal conv layer to decrease the depth
        self.conv_out = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x):
        """
        Feed Forward function.
        :param x: Input data
        :return: x: output prediction
        """
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # add third hidden layer
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # add fourth hidden layer
        # x = F.relu(self.conv4(x))
        # x = self.pool(x) # compressed representation

        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        # x = F.relu(self.t_conv4(x))
        # transpose again, output should have a sigmoid applied
        x = F.relu(self.conv_out(x))

        return x

