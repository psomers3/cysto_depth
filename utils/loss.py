import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils.rendering import PhongRender
from config.training_config import PhongConfig
from typing import *


class CosineSimilarity(nn.Module):
    def __init__(self, ignore_direction: bool = False, device: torch.device = None):
        """
        TODO: WARNING!!! ignore_direction may break gradient... needs to be looked into
        :param ignore_direction:
        :param device:
        """
        super(CosineSimilarity, self).__init__()
        self.loss = torch.nn.CosineSimilarity(dim=1)
        self.device = device
        self.ignore_direction = ignore_direction

    def forward(self, predicted, target) -> torch.Tensor:
        predicted = F.normalize(predicted, dim=1)
        non_zero_norm = torch.linalg.norm(target, dim=1) > 0.0
        if self.ignore_direction:
            epsilon = 1e-6
            return (1 - torch.where(non_zero_norm, self.loss(predicted, target)+1+epsilon,
                                    torch.ones([1], device=self.device)).abs()).mean()
        else:
            return (1 - torch.where(non_zero_norm, self.loss(predicted, target),
                                    torch.ones([1], device=self.device))).mean()


class BerHu(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerHu, self).__init__()
        self.threshold = threshold

    def forward(self, predicted, target):
        # fill background with 
        if not predicted.shape == target.shape:
            _, _, h, w = target.shape
            predicted = F.interpolate(predicted, size=(h, w), mode='bilinear', align_corners=True)
        # mask = target > 0
        # predicted = predicted * mask
        diff = torch.abs(target - predicted)
        delta = self.threshold * torch.max(diff).item()

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff ** 2. - delta ** 2, 0., -delta ** 2.) + delta ** 2.
        part2 = part2 / (2. * delta)

        loss = part1 + part2
        # loss = torch.sum(loss)
        loss = torch.mean(loss)
        return loss


class GradientLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1.weight = nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0))
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2.weight = nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0))
        # self.silog = SILog()

    def forward(self, predicted, target):
        if not predicted.shape == target.shape:
            _, _, H, W = target.shape
            predicted = F.interpolate(predicted, size=(H, W), mode='bilinear', align_corners=True)
        p_x, p_y = self.get_gradient(predicted)
        t_x, t_y = self.get_gradient(target)
        dy = p_y - t_y
        dx = p_x - t_x
        # si_loss = self.silog(predicted, target)
        grad_loss = torch.mean(torch.pow(dx, 2) + torch.pow(dy, 2))
        return grad_loss

    def get_gradient(self, x):
        G_x = self.conv1(Variable(x))
        G_y = self.conv2(Variable(x))
        return G_x, G_y


class PhongLoss(nn.Module):
    def __init__(self, config: PhongConfig, image_size: int = 256, device=None) -> None:
        super(PhongLoss, self).__init__()
        self.phong_renderer = PhongRender(config=config, image_size=image_size, device=device)
        self.image_loss = torch.nn.MSELoss()

    def forward(self, predicted_depth_normals: Tuple[torch.Tensor, ...], true_phong: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param predicted_depth_normals:
        :param true_phong:
        :return: the loss value and the rendered images
        """
        rendered = self.phong_renderer(predicted_depth_normals)
        return self.image_loss(rendered, true_phong), rendered


class AvgTensorNorm(nn.Module):
    def forward(self, predicted):
        avg_norm = torch.norm(predicted, p='fro')
        return avg_norm


def binary_cross_entropy_loss(predicted: Tensor, ground_truth: Union[Tensor, float], *args, **kwargs) -> Tensor:
    if isinstance(ground_truth, float):
        ground_truth = torch.ones_like(predicted, device=predicted.device)
    return F.binary_cross_entropy(predicted, ground_truth)


def wasserstein_discriminator_loss(original: Tensor, generated: Tensor, *args, **kwargs) -> Tensor:
    """ Discriminator outputs from data originating from the original domain and
        from the generator's attempt (generated)
    """
    return generated.mean() - original.mean()


def wasserstein_generator_loss(generated: Tensor, *args, **kwargs) -> Tensor:
    return -generated.mean()


def wasserstein_gradient_penalty(original_input: Tensor,
                                 generated_input: Tensor,
                                 critic: torch.nn.Module,
                                 wasserstein_lambda: float = 10) -> Tensor:
    batch_size = original_input.shape[0]
    epsilon = torch.rand(batch_size, *[1] * (original_input.ndim - 1), device=original_input.device)

    interpolated_img = epsilon * original_input + (1 - epsilon) * generated_input
    interpolated_out = critic(interpolated_img)

    grads = torch.autograd.grad(outputs=interpolated_out, inputs=interpolated_img,
                                grad_outputs=torch.ones_like(interpolated_out, device=original_input.device),
                                create_graph=True, retain_graph=True)[0]
    grads = grads.reshape([batch_size, -1])
    grad_penalty = wasserstein_lambda * ((grads.norm(2, dim=1) - 1) ** 2).mean()  # take norm per batch
    return grad_penalty


def wasserstein_gp_discriminator_loss(original_input: Tensor,
                                      generated_input: Tensor,
                                      critic: torch.nn.Module,
                                      wasserstein_lambda: float = 10) -> Tensor:
    """ https://arxiv.org/abs/1704.00028 """
    original_input.requires_grad = True
    generated_input.requires_grad = True

    return (wasserstein_discriminator_loss(critic(original_input), critic(generated_input)) \
           + wasserstein_gradient_penalty(original_input, generated_input, critic, wasserstein_lambda)) * 1e-3


GANDiscriminatorLoss: Dict[str, Callable[..., Tensor]] = {
    'wasserstein_gp': wasserstein_gp_discriminator_loss,
    'cross_entropy': binary_cross_entropy_loss,
    'wasserstein': wasserstein_discriminator_loss}
GANGeneratorLoss: Dict[str, Callable[..., Tensor]] = {
    'wasserstein': wasserstein_generator_loss,
    'cross_entropy': binary_cross_entropy_loss,
    'wasserstein_gp': wasserstein_generator_loss}
