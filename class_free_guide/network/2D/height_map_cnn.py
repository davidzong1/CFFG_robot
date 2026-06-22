import torch
import torch.nn as nn
from class_free_guide.network.base.cnn.conv2d import Conv2dBlock, Conv2dNet


class HeightMapCNN3Layer(nn.Module):
    def __init__(self, input_channels=1, output_dim=128):
        super(HeightMapCNN3Layer, self).__init__()
        self.CNN = Conv2dNet(
            input_channels,
            [32, 64, 128],
            kernel_sizes=[3, 3, 3],
            strides=[1, 1, 1],
            paddings=[1, 1, 1],
            activations=["relu", "relu", "relu"],
        )
        self.fc = nn.Linear(128 * 8 * 8, output_dim)  #

    def forward(self, x):
        x = self.CNN(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc(x))
        return x
