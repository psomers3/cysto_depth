from torch import nn
import torch
from utils.torch_utils import convrelu


class ImgDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv = nn.Sequential(
            # Shape N, 512, x.H/32, x.W/32
            # receptive field: 
            convrelu(in_channels, 64, 4, 0, 2, norm="instance", relu="leaky", alpha=0.2),
            torch.nn.Dropout(),
            convrelu(64, 128, 4, 1, 2, norm="instance", relu="leaky", alpha=0.2),
            torch.nn.Dropout(),
            convrelu(128, 256, 4, 1, 2, norm="instance", relu="leaky", alpha=0.2),
            torch.nn.Dropout(),
            convrelu(256, 512, 4, 1, 2, norm="instance", relu="leaky", alpha=0.2),
            torch.nn.Dropout(),
            nn.Conv2d(512, 1, 4, 1, 2),
            nn.Flatten()
        )
        self.out = nn.Sequential(
            nn.Sigmoid()
        )

    def forward(self, _input):
        validity = self.conv(_input)
        validity = self.out(validity)
        return validity

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return super(ImgDiscriminator, self).__call__(*args, **kwargs)
