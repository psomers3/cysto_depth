from utils.image_utils import create_circular_mask
import torch
from torchvision import transforms as torch_transforms
from typing import *


class SynchronizedTransform:
    """
    A helper class to synchronize pytorch transformations used in separate transformation compositions. Ideal for
    things like color images and segmentation labels.
    """

    def __init__(self,
                 transform: Callable,
                 num_synchros=2,
                 additional_args: List[List[Any]] = None):
        self.transform = transform
        self.num_synchros = num_synchros
        self._generator_state = torch.get_rng_state()
        self._sync_count = set()
        self.additional_args = additional_args if additional_args else [[] for _ in range(num_synchros)]

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        current_gen_state = torch.get_rng_state()
        if len(self._sync_count) < self.num_synchros:
            torch.set_rng_state(self._generator_state)
        else:
            self._sync_count.clear()
            self._generator_state = torch.get_rng_state()
        transformed = self.transform(data, *self.additional_args[len(self._sync_count)])
        self._sync_count.add(len(self._sync_count))
        torch.set_rng_state(current_gen_state)
        return transformed


class RandomAffine:
    def __init__(self, degrees: Tuple[float, float], translate: Tuple[float, float]):
        self.degrees = degrees
        self.translate = translate

    def __call__(self, data: torch.Tensor, use_corner_as_fill: bool = False) -> torch.Tensor:
        border_color = torch.mean(data[:, [0, -1, 0, 1], [0, -1, 0, 1]], dim=-1)
        fill = border_color.tolist() if use_corner_as_fill else 0
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
            radius = (self.radius_factor[1]-self.radius_factor[0]) * randomized_radius + self.radius_factor[0]

        data[:, create_circular_mask(*data.shape[-2:], invert=True, radius_scale=radius)] = mask_color
        return data


class Squarify:
    """
    A transform to center crop an image to be a square using the shorter image dimension. If image size is provided, the
    square image is then resized to the provided dimension.
    """
    def __init__(self, image_size: int = None):
        self.image_size = image_size
        self.resize = None
        if self.image_size is not None:
            self.resize = torch_transforms.Resize(self.image_size)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        data = torch_transforms.CenterCrop(min(data.shape[-2:]))(data)
        if self.image_size is not None:
            data = self.resize(data)
        return data