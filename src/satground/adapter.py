import torch
from torch import nn


class StructuralAdapter(nn.Module):
    """Shared residual adaptation of pre-render scene features, never of target RGB."""
    def __init__(self, channels=64, width=16):
        super().__init__()
        self.channels, self.width = channels, width
        self.net = nn.Sequential(nn.Conv2d(channels, width, 1), nn.SiLU(),
                                 nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(width, channels, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return features + self.net(features)
