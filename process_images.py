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
import cv2


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


def get_normals_from_depthmap(depth_map: np.ndarray) -> np.ndarray:
    # zy, zx = np.gradient(depth_map)
    # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
    # to reduce noise
    zx = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=5)
    zy = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=5)

    normal = np.dstack((-zx, -zy, np.ones_like(depth_map)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n
    return normal


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_ckpt', type=str, help='path to model checkpoint')
    parser.add_argument('--img_dir', type=str, help='path to the directory containing images to process')
    parser.add_argument('--output_dir', type=str, default=None, help='output directory to save numpy arrays. Defaults'
                                                                     ' to no output.')
    parser.add_argument('--view_results', action='store_true', help='whether to display the resulting depth and normals')
    args = parser.parse_args()
    results = process_directory(args.img_dir, args.model_ckpt)
    if not args.output_dir:
        exit(0)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for img_path, d_map in results.items():
        name = os.path.splitext(os.path.basename(img_path))[0]
        np.save(os.path.join(args.output_dir, name), d_map)
        normal = get_normals_from_depthmap(d_map)
        np.save(f'{os.path.join(args.output_dir, name)}_normals', normal)

        cv2.imshow('color', cv2.imread(img_path))
        if args.view_results:
            d_img = d_map - d_map.min()
            d_img /= d_img.max()
            d_img *= 255
            cv2.imshow('depth', d_img.astype(np.uint8))

            # offset and rescale values to be in 0-255
            normal += 1
            normal /= 2
            normal *= 255

            cv2.imshow("normal", normal[:, :, ::-1].astype(np.uint8))
            cv2.waitKey(0)
