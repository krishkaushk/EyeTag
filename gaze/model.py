# Defines the neural network that maps eye features to screen coordinates.


import torch      
import torch.nn as nn   


class GazeNet(nn.Module):
    # Fully connected nn
    

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(22, 256),
            nn.GELU(),

            nn.Linear(256, 256),
            nn.GELU(),

            nn.Linear(256, 128),
            nn.GELU(),

            nn.Linear(128, 64),
            nn.GELU(),

            nn.Linear(64, 32),
            nn.GELU(),

            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)
