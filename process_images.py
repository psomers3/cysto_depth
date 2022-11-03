import os
import numpy as np
from typing import *
import argparse
from pathlib import Path
import imghdr
from models.gan_model import GAN
from torchvision.io.image import read_image
import torch
from torchvision import transforms

image_types = ['png', 'jpeg']
image_float_transform = transforms.ConvertImageDtype(torch.float)


def process_directory(directory: str, model_ckpt: str) -> Dict[str, np.ndarray]:
    """
    Load a GAN depth prediction model from a given checkpoint and use it to process all photos in a provided directory

    :param directory: directory of photos
    :param model_ckpt: the model checkpoint to load
    :return: dictionary of depth maps with keys corresponding to the source image path
    """
    model: GAN = GAN.load_from_checkpoint(model_ckpt, res_transfer=True, adaptive_gating=True)
    image_paths = [f for f in Path(directory).rglob('*') if imghdr.what(f) in image_types]
    depth_maps = {}
    for f in image_paths:
        f = str(f)
        depth_map = model.forward(image_float_transform(read_image(f))[None], full_prediction=True)[-1].detach().numpy()
        depth_maps[f] = np.squeeze(np.squeeze(depth_map, axis=0))
    return depth_maps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_ckpt', type=str, help='path to model checkpoint')
    parser.add_argument('--img_dir', type=str, help='path to the directory containing images to process')
    parser.add_argument('--output_dir', type=str, default=None, help='output directory to save numpy arrays. Defaults'
                                                                     ' to no output.')
    args = parser.parse_args()
    results = process_directory(args.img_dir, args.model_ckpt)
    if not args.output_dir:
        exit(0)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for img_path, depth_map in results.items():
        name = os.path.splitext(os.path.basename(img_path))[0]
        np.save(os.path.join(args.output_dir, name), depth_map)
