from exr_utils import create_circular_mask
import matplotlib.pyplot as plt
from torchvision.transforms import GaussianBlur
import torch
import torch.nn.functional as F
from pathlib import Path
from torchvision.io import read_image

p = f'/Users/peter/isys/2023_01_25/color/bladder_wall'

files = list(Path(p).glob('*'))
img: torch.Tensor = read_image(str(files[0]))/255
kernel_size = 101
blur = GaussianBlur(kernel_size=kernel_size, sigma=30)
mask = torch.Tensor(create_circular_mask(*img.shape[-2:]))[None]
pad_size = kernel_size//2 + 1
mask = F.pad(mask, [pad_size]*4, 'constant', 0)
blurred_mask = blur(mask)[:, pad_size:-pad_size, pad_size:-pad_size]
mask = mask[:, pad_size:-pad_size, pad_size:-pad_size]
blurred_img = img - 1 + blurred_mask*mask
plt.imshow(blurred_img.permute([1, 2, 0]))
plt.show(block=True)