# Defines the neural network that maps eye features to screen coordinates.


import torch      
import torch.nn as nn   


class GazeNet(nn.Module):
    # Fully connected nn
    

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # --- LAYER 1 ---
            nn.Linear(22, 64),

            # --- ACTIVATION 1 ---
            nn.ReLU(),

            # --- LAYER 2 ---
            nn.Linear(64, 32),

            # --- ACTIVATION 2 ---
            nn.ReLU(),

            # --- OUTPUT LAYER ---           
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)
