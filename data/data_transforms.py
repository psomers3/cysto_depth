import torchvision.transforms

from utils.image_utils import create_circular_mask
import numpy as np
import torch
from torchvision import transforms as torch_transforms
from torchvision.transforms import functional as torch_transforms_func
from typing import *


class SynchronizedTransform:
    """
    A helper class to synchronize pytorch transformations that are randomized to be used in separate transformation
     compositions. Ideal for things like color images and segmentation labels.
    """

    def __init__(self,
                 transform: Callable,
                 num_synchros=2,
                 additional_args: List[List[Any]] = None):
        """

        :param transform: The transform to be applied
        :param num_synchros: how many items will be getting this transform before advancing the randomness
        :param additional_args: A list of arguments that will be passed to the transform for each synchronized item.
                                There should be the same number of items in this list as num_synchros.
        """
        self.transform = transform
        self.num_synchros = num_synchros
        self._generator_state = torch.get_rng_state()
        self._sync_count = 0
        self.additional_args = additional_args if additional_args else [[] for _ in range(num_synchros)]

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        current_gen_state = torch.get_rng_state()
        if self._sync_count < self.num_synchros:
            torch.set_rng_state(self._generator_state)
        else:
            self._sync_count = 0
            self._generator_state = torch.get_rng_state()
        transformed = self.transform(data, *self.additional_args[self._sync_count])
        self._sync_count += 1
        torch.set_rng_state(current_gen_state)
        return transformed


class RandomAffine:
    def __init__(self, degrees: Tuple[float, float], translate: Tuple[float, float], use_corner_as_fill: bool = None):
        self.degrees = degrees
        self.translate = translate
        self.use_corner_as_fill = use_corner_as_fill

    def __call__(self, data: torch.Tensor, use_corner_as_fill: bool = False) -> torch.Tensor:
        border_color = torch.mean(data[:, [0, -1, 0, 1], [0, -1, 0, 1]], dim=-1)
        if self.use_corner_as_fill is None:
            fill = border_color.tolist() if use_corner_as_fill else 0
        else:
            fill = border_color.tolist() if self.use_corner_as_fill else 0
        affine = torch_transforms.RandomAffine(degrees=self.degrees, translate=self.translate, fill=fill)
        return affine(data)


class EndoMask:
    """
    A transform that will apply a mask that covers everything with the mask color except a circle in the center
    """

    def __init__(self,
                 mask_color: Union[float, List[float]] = None,
                 radius_factor: Union[float, List[float]] = 1.0):
        """
        :param mask_color: color to use for mask. If left as none, a randomized dark color is used per image.
        :param radius_factor: the circle radius will be made to this factor of 1/2 the minimum image dimension. If  a
                              range is given [min, max], the radius is randomly chosen from this range per image.
        """
        self.mask_color = mask_color
        self.radius_factor = radius_factor

    def __call__(self, data: torch.Tensor, mask_color: Any = None) -> torch.Tensor:
        randomized_color = torch.rand((3, 1), dtype=torch.float) / 10
        randomized_radius = torch.rand(1, dtype=torch.float).numpy()

        if mask_color is None:
            if self.mask_color is None:
                mask_color = randomized_color
            else:
                mask_color = self.mask_color
        if isinstance(self.radius_factor, float):
            radius = self.radius_factor
        else:
            radius = (self.radius_factor[1] - self.radius_factor[0]) * randomized_radius + self.radius_factor[0]

        data[:, create_circular_mask(*data.shape[-2:], invert=True, radius_scale=radius)] = mask_color
        return data


class Squarify:
    """
    A transform to center crop an image to be a square using the shorter image dimension. If image size is provided, the
    square image is then resized to the provided dimension.
    """

    def __init__(self, image_size: int = None, clamp_values: bool = False):
        self.image_size = image_size
        self.resize = None
        self.clamp = clamp_values
        if self.image_size is not None:
            self.resize = torch_transforms.Resize(self.image_size,
                                                  interpolation=torchvision.transforms.InterpolationMode.BICUBIC)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        data = torch_transforms.CenterCrop(min(data.shape[-2:]))(data)
        if self.image_size is not None:
            data = self.resize(data)
        if self.clamp:
            return torch.clamp(data, 0.0, 1.0)
        else:
            return data

class TensorSlice:
    def __init__(self, torch_slice: tuple):
        self.slice = torch_slice

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data[self.slice][None]


class ElementWiseScale:
    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data * self.factor


class AddGaussianNoise:
    def __init__(self, mean=0., std=1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return torch.clamp(tensor + torch.randn(tensor.size()) * self.std + self.mean, min=0, max=1)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class PhongAffine:
    """ special rotation to apply the proper rotation to normals """

    def __init__(self,
                 degrees: Tuple[float, float],
                 translate: Tuple[float, float],
                 use_corner_as_fill: bool = None,
                 image_size: int = 256,
                 device: torch.device = None):
        """
        TODO: implement translation properly... for now DO NOT TRANSLATE
        :param degrees:
        :param translate:
        :param use_corner_as_fill:
        :param image_size:
        """
        self.degrees = degrees
        self.translate = translate
        self.use_corner_as_fill = use_corner_as_fill
        self.image_size = (image_size, image_size)
        self.device = device

    def __call__(self, data: torch.Tensor, use_corner_as_fill: bool = False, is_normals: bool = False) -> torch.Tensor:
        border_color = torch.mean(data[:, [0, -1, 0, 1], [0, -1, 0, 1]], dim=-1)
        if self.use_corner_as_fill is None:
            fill = border_color.tolist() if use_corner_as_fill else 0
        else:
            fill = border_color.tolist() if self.use_corner_as_fill else 0
        degrees, translation, scale, shear = torch_transforms.RandomAffine.get_params(degrees=self.degrees,
                                                                                      translate=self.translate,
                                                                                      img_size=self.image_size,
                                                                                      scale_ranges=None,
                                                                                      shears=None)
        if data.shape[0] == 3 and is_normals:
            radians = np.deg2rad(degrees)
            rotation_matrix = torch.Tensor([[np.cos(radians), np.sin(radians), 0],
                                            [-np.sin(radians), np.cos(radians), 0],
                                            [0, 0, 1]]).to(self.device)
            permuted = data.permute((1, 2, 0))
            normals_reshaped = permuted.reshape((data.shape[1] * data.shape[2], 3))
            rotated = (rotation_matrix[None] @ normals_reshaped.unsqueeze(-1)).squeeze(-1)
            data = rotated.permute((1, 0)).reshape(data.shape)

        transformed = torch_transforms_func.affine(data, degrees, translation, scale, shear, fill=fill)
        return transformed
