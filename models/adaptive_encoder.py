from utils.loss import AvgTensorNorm
from models.vanillaencoder import VanillaEncoder
from utils.torch_utils import convrelu
from torch import nn
import torch
from config.training_config import EncoderConfig


class AdaptiveEncoder(VanillaEncoder):
    def __init__(self, config: EncoderConfig):
        """

        :param config: Config for the Encoder
        """
        super().__init__(backbone=config.backbone, imagenet_weights=config.load_imagenet_weights)
        init_zero = False
        activation = config.res_layer_activation
        norm = config.res_layer_norm
        self.adaptive_gating = config.adaptive_gating
        self.residual_learning = config.residual_learning

        if not self.residual_learning:
            # pure vanilla case
            return

        for param in self.parameters():
            param.requires_grad = False

        self.gate_coefficients = nn.Parameter(torch.zeros(5), requires_grad=self.adaptive_gating)

        self.res_layer0 = nn.Sequential(convrelu(3, self.feature_levels[0], 5, 2, 1,
                                                 activation=activation, norm=norm, init_zero=init_zero),
                                        convrelu(self.feature_levels[0], 3, 5, 2, 1,
                                                 activation=activation, init_zero=init_zero, norm=norm))
        self.res_layer1 = nn.Sequential(convrelu(self.feature_levels[0], self.feature_levels[1], 5, 2, 1,
                                                 activation=activation, norm=norm, init_zero=init_zero),
                                        convrelu(self.feature_levels[1], self.feature_levels[0], 5, 2, 1,
                                                 activation=activation, init_zero=init_zero, norm=norm))
        self.res_layer2 = nn.Sequential(convrelu(self.feature_levels[1], self.feature_levels[2], 5, 2, 1,
                                                 activation=activation, norm=norm, init_zero=init_zero),
                                        convrelu(self.feature_levels[2], self.feature_levels[1], 5, 2, 1,
                                                 activation=activation, init_zero=init_zero, norm=norm))
        self.res_layer3 = nn.Sequential(
            convrelu(self.feature_levels[2], self.feature_levels[3], 3, 1, 1,
                     activation=activation, norm=norm, init_zero=init_zero),
            convrelu(self.feature_levels[3], self.feature_levels[2], 3, 1, 1,
                     activation=activation, init_zero=init_zero, norm=norm)
        )
        self.res_layer4 = nn.Sequential(
            convrelu(self.feature_levels[3], self.feature_levels[4], 3, 1, 1,
                     activation=activation, norm=norm, init_zero=init_zero),
            convrelu(self.feature_levels[4], self.feature_levels[3], 3, 1, 1,
                     activation=activation, init_zero=init_zero, norm=norm)
        )
        self.criterion = AvgTensorNorm()

    def set_residuals_train(self):
        """ helper function to turn on batch norm updates after turning the rest of them in the network off. """
        if self.residual_learning:
            self.res_layer0.train()
            self.res_layer1.train()
            self.res_layer2.train()
            self.res_layer3.train()
            self.res_layer4.train()

    def set_residuals_eval(self):
        """ helper function to turn off batch norm updates after turning the rest of them in the network on. """
        if self.residual_learning:
            self.res_layer0.eval()
            self.res_layer1.eval()
            self.res_layer2.eval()
            self.res_layer3.eval()
            self.res_layer4.eval()

    def _gate(self, input_tensor1: torch.Tensor, input_tensor2: torch.Tensor, level: int) -> torch.Tensor:
        if self.adaptive_gating:
            return input_tensor1 + torch.tanh(self.gate_coefficients[level]) * input_tensor2
        else:
            return input_tensor1 + input_tensor2

    def forward(self, encoder_input):
        if not self.residual_learning:
            return VanillaEncoder.forward(self, encoder_input)
        x_original = self.conv_original_size0(encoder_input)
        x_original = self.conv_original_size1(x_original)

        res_layer0 = self.res_layer0(encoder_input)
        layer0 = self.layer0(self._gate(encoder_input, res_layer0, 0))

        res_layer1 = self.res_layer1(layer0)
        layer1 = self.layer1(self._gate(layer0, res_layer1, 1))

        res_layer2 = self.res_layer2(layer1)
        layer2 = self.layer2(self._gate(layer1, res_layer2, 2))

        res_layer3 = self.res_layer3(layer2)
        layer3 = self.layer3(self._gate(layer2, res_layer3, 3))

        res_layer4 = self.res_layer4(layer3)
        layer4 = self.layer4(self._gate(layer3, res_layer4, 4))

        layer0_mare = self.criterion(res_layer0)
        layer1_mare = self.criterion(res_layer1)
        layer2_mare = self.criterion(res_layer2)
        layer3_mare = self.criterion(res_layer3)
        layer4_mare = self.criterion(res_layer4)

        layer0 = self.layer0_skip(layer0)
        layer1 = self.layer1_skip(layer1)
        layer2 = self.layer2_skip(layer2)
        layer3 = self.layer3_skip(layer3)
        layer4 = self.layer4_skip(layer4)
        return ([x_original, layer0, layer1, layer2, layer3, layer4],
                [layer0_mare, layer1_mare, layer2_mare, layer3_mare, layer4_mare])


class ConditionalMeanRelativeLoss(nn.Module):
    def __init__(self):
        super(ConditionalMeanRelativeLoss, self).__init__()

    def forward(self, output, target):
        # calculate absolute errors
        absolute_output = torch.abs(output)
        # where target is too small, use just the absolute errors to avoid divide by 0
        # but clamp abs (target) away from zero to avoid "ghost" divide by 0
        abs_target = torch.abs(target).clamp(0.0005)
        loss = torch.where(abs_target < 0.001, absolute_output, torch.divide(absolute_output, abs_target))
        # return mean loss
        return torch.mean(loss)
