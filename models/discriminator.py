from torch import nn
import torch
from utils.torch_utils import convrelu
from config.training_config import DiscriminatorConfig


_reductions = {'sum': torch.sum,
               'min': torch.min,
               'max': torch.max,
               'mean': torch.mean}


class Discriminator(nn.Module):
    def __init__(self, config: DiscriminatorConfig):
        super().__init__()
        self.single_out = config.single_out
        activation = config.activation
        norm = config.normalization
        in_channels = config.in_channels
        self.use_sigmoid = config.use_sigmoid
        self.reduction = _reductions[config.single_out_reduction.lower()]

        if config.img_level:
            self.conv = nn.Sequential(
                # Shape N, 512, x.H/32, x.W/32
                # receptive field:
                convrelu(in_channels, 64, 4, 0, 2, norm=norm, activation=activation, alpha=0.2),
                torch.nn.Dropout(),
                convrelu(64, 128, 4, 1, 2, norm=norm, activation=activation, alpha=0.2),
                torch.nn.Dropout(),
                convrelu(128, 256, 4, 1, 2, norm=norm, activation=activation, alpha=0.2),
                torch.nn.Dropout(),
                convrelu(256, 512, 4, 1, 2, norm=norm, activation=activation, alpha=0.2),
                torch.nn.Dropout(),
                nn.Conv2d(512, 1, 4, 1, 2),
                nn.Flatten()
            )
        else:
            self.conv = nn.Sequential(
                # Shape N, 512, x.H/32, x.W/32
                convrelu(in_channels, 64, 4, 1, 2, alpha=0.2),
                torch.nn.Dropout(),
                convrelu(64, 128, 4, 1, 2, norm=norm, activation=activation, alpha=0.2),
                torch.nn.Dropout(),
                convrelu(128, 256, 3, 1, 1, norm=norm, activation=activation, alpha=0.2),
                torch.nn.Dropout(),
                convrelu(256, 512, 3, 1, 1, norm=norm, activation=activation, alpha=0.2),
                nn.Conv2d(512, 1, 3, 1, 1),
                nn.Flatten()
            )
        self.sigmoid = nn.Sequential(
            nn.Sigmoid()
        )

    def forward(self, _input):
        validity = self.conv(_input)
        if self.single_out:
            validity = self.reduction(validity)
        if self.use_sigmoid:
            validity = self.sigmoid(validity)
        return validity
