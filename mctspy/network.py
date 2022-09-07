"""
Neural Network
"""

import numpy as np
import torch
import torch.nn as nn

from parameters import params

class A2CNet(nn.Module):
    """
    A2C Neural Network
    """

    def __init__(self) -> None:
        super().__init__()

        net = [
            nn.Conv1d(3, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        ]

        for _ in range(params.convolution_layers - 3):
            net += [
                nn.Conv1d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
            ]

        net += [
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=768, out_features=512),
            nn.ReLU(inplace=True),
        ]

        self.net = nn.Sequential(*net)

        self.val = nn.Sequential(
            nn.Linear(in_features = 512, out_features = 512),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 512, out_features = 256),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 256, out_features = 1),
            nn.Tanh()
        )

        self.pol = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(inplace = True),
            nn.Linear(512, 256), nn.ReLU(inplace = True),
            nn.Linear(256, 9),
            nn.LogSoftmax(dim=-1),
        )


    def forward(self, X):
        """
        Equivalent of a .fit for a neural network in pytorch
        """
        y = self.net(X)
        y_val = self.val(y)
        y_pol = self.pol(y).double() # We need double precision for small probabilities.
        # torch.clamp(y_pol, min=10e-39, max=1)
        return y_val, y_pol.exp()


# Tests
example_tuple = torch.tensor(
    np.array([[-1.,  1.,  1.],
       [ 1., -1., -1.],
       [-1.,  1.,  1.]])
    )
example_tuple = torch.unsqueeze(example_tuple, dim = -1).float()
example_tuple = torch.permute(example_tuple.float(), (2, 0, 1))

A2C = A2CNet()
A2C(example_tuple)
print(example_tuple, example_tuple.shape)
