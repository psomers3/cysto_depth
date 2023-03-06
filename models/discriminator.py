from torch import nn
import torch
from utils.torch_utils import convrelu
from config.training_config import DiscriminatorConfig


class Sum(nn.Module):
    def forward(self, x):
        return torch.sum(x)


class Min(nn.Module):
    def forward(self, x):
        return torch.min(x)


class Max(nn.Module):
    def forward(self, x):
        return torch.max(x)


class Mean(nn.Module):
    def forward(self, x):
        return torch.mean(x)


_reductions = {'sum': Sum,
               'min': Min,
               'max': Max,
               'mean': Mean,
               'dense': torch.nn.Linear}

_output_activation = {'sigmoid': torch.sigmoid,
                      'tanh': torch.tanh}


class Discriminator(nn.Module):
    def __init__(self, config: DiscriminatorConfig):
        super().__init__()
        self.single_out = config.single_out
        activation = config.activation
        norm = config.normalization
        in_channels = config.in_channels
        self.output_activation = config.output_activation.lower() if config.output_activation else None
        self.reduction = config.single_out_reduction.lower()
        self._linear = None

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
        if self.single_out and (self.reduction != 'dense'):
            self.conv.append(_reductions[self.reduction]())

    def forward(self, _input):
        validity = self.conv(_input)
        if self.single_out and self.reduction == 'dense':
            if self._linear is None:
                self._linear = torch.nn.Linear(validity.shape[1], 1, device=validity.device)
            validity = self._linear(validity)
        if self.output_activation is not None:
            validity = _output_activation[self.output_activation](validity)
        return validity

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return super(Discriminator, self).__call__(*args, **kwargs)
