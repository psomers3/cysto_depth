from torch import nn
import torch
from utils.torch_utils import convrelu


class Discriminator(nn.Module):
    def __init__(self, in_channels, single_out: bool = False):
        super().__init__()
        self.single_out = single_out

        self.conv = nn.Sequential(
            # Shape N, 512, x.H/32, x.W/32
            convrelu(in_channels, 64, 4, 1, 2, alpha=0.2),
            torch.nn.Dropout(),
            convrelu(64, 128, 4, 1, 2, norm="instance", activation="leaky", alpha=0.2),
            torch.nn.Dropout(),
            convrelu(128, 256, 3, 1, 1, norm="instance", activation="leaky", alpha=0.2),
            torch.nn.Dropout(),
            convrelu(256, 512, 3, 1, 1, norm="instance", activation="leaky", alpha=0.2),
            nn.Conv2d(512, 1, 3, 1, 1),
            nn.Flatten()
        )
        self.out = nn.Sequential(
            nn.Sigmoid()
        )

    def forward(self, input):
        validity = self.conv(input)
        if self.single_out:
            validity = validity.max()
        validity = self.out(validity)
        return validity