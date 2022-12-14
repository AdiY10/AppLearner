import torch.nn as nn
import torch.optim as optim
import numpy as np

"""
creating CNN for time series prediction
"""

class CNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN, self).__init__()
        self.__seq_model = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=20*2, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=output_size)
        )

    def forward(self, x):
        out = self.__seq_model(x)
        return out

    def flatten_parameters(self):
        pass



