import os
import hydra
from pathlib import Path
import torch
import numpy as np
import re
import cv2 as cv
from omegaconf import OmegaConf
from config.training_config import CystoDepthConfig
from simple_parsing import ArgumentParser
from typing import *
from tqdm.contrib import tzip
from isys_optitrack.image_tools import get_circular_mask_4_img, ImageCroppingException
from models.gan_model import GAN
from utils.exr_utils import crop_img_opencv
from data.data_transforms import ImageNetNormalization

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm


class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val: np.ndarray):
        return self.scalarMap.to_rgba(val)


imagenet_norm = ImageNetNormalization()

recording_directory = ""


@hydra.main(version_base=None, config_path="config", config_name="training_config")
def cysto_depth(cfg: CystoDepthConfig) -> None:
    config: Union[Any, CystoDepthConfig] = OmegaConf.merge(OmegaConf.structured(CystoDepthConfig()), cfg, )
    if config.print_config:
        print(OmegaConf.to_yaml(config))

    torch.manual_seed(config.global_seed)
    model = GAN(synth_config=config.synthetic_config.copy(), gan_config=config.gan_config.copy())

    directory = recording_directory
    depth_directory = os.path.join(directory, 'peter_depth_rendering')
    depth_imgs = [os.path.join(depth_directory, x) for x in
                  sorted(os.listdir(depth_directory), key=lambda k: re.findall('.(\d+).', k)) if x[-3:] == 'exr']
    images = [os.path.join(depth_directory, x) for x in
              sorted(os.listdir(depth_directory), key=lambda k: re.findall('.(\d+).', k)) if x[-3:] == 'png']
    depth_max = 0.075
    skip = 1
    start_frame = None
    if start_frame is not None:
        frame_start = start_frame
    else:
        frame_start = int(re.findall('.(\d+).', os.path.basename(depth_imgs[0]))[0])
    with_mask = True
    video_path = Path(directory, 'video.mp4')
    render_path = Path(directory, "presentation", 'rendering.mp4')
    render_path.parent.mkdir(parents=True, exist_ok=True)
    render_depth_path = Path(directory, "presentation", 'rendering_depth.mp4')
    video_rendered_img_path = Path(directory, "presentation", 'video_with_img.mp4')
    depth_prediction_path = Path(directory, "presentation", 'depth_prediction.mp4')
    color_prediction_path = Path(directory, "presentation", 'color_prediction.mp4')
    diff_path = Path(directory, "presentation", 'difference.mp4')
    phong_path = Path(directory, "presentation", 'phong.mp4')
    prediction_path = Path(directory, "presentation", 'prediction.mp4')
    color_path = Path(directory, "presentation", 'color.mp4')
    depth_path = Path(directory, "presentation", 'depth.mp4')

    cap = cv.VideoCapture(str(video_path))
    fps = int(20 / skip)
    video_codec = cv.VideoWriter_fourcc(*'mp4v')

    render_writer = cv.VideoWriter(str(render_path),
                                  video_codec,
                                  fps, (1920 * 1, 1080))
    render_depth_writer = cv.VideoWriter(str(render_depth_path),
                                   video_codec,
                                   fps, (1920 * 1, 1080))
    img_writer = cv.VideoWriter(str(video_rendered_img_path),
                                video_codec,
                                fps, (1920 * 2, 1080))
    depth_prediction_writer = cv.VideoWriter(str(depth_prediction_path),
                                             video_codec,
                                             fps, (256 * 3, 256))

    color_prediction_write = cv.VideoWriter(str(color_prediction_path),
                                            video_codec,
                                            fps, (256 * 2, 256))

    diff_writer = cv.VideoWriter(str(diff_path),
                                 video_codec,
                                 fps, (256 * 1, 256))

    phong_writer = cv.VideoWriter(str(phong_path),
                                  video_codec,
                                  fps, (256 * 1, 256))
    prediction_writer = cv.VideoWriter(str(prediction_path),
                                       video_codec,
                                       fps, (256 * 1, 256))
    color_writer = cv.VideoWriter(str(color_path),
                                  video_codec,
                                  fps, (256 * 1, 256))
    depth_writer = cv.VideoWriter(str(depth_path),
                                  video_codec,
                                  fps, (256 * 1, 256))

    print(f'number of frames: {len(depth_imgs)}')
    good = False
    i = 0
    last_mask = None
    if with_mask:
        while not good and last_mask is None:
            good, frame = cap.read()
            if good:
                good = False
                try:
                    last_mask = np.expand_dims(get_circular_mask_4_img(frame), -1)
                except ImageCroppingException:
                    pass
                continue
        cap.release()
        cap = cv.VideoCapture(os.path.join(directory, 'video.mp4'))
    cap.set(1, frame_start - 1)

    for depth_img, image in tzip(depth_imgs, images):
        while not good and i < len(depth_imgs) and last_mask is not None:
            good, frame = cap.read()
            if good and i % skip != 0:
                good = False
                i += 1
                continue
            if good:
                try:
                    last_mask = np.expand_dims(get_circular_mask_4_img(frame), -1)
                except ImageCroppingException:
                    pass
                continue
        i += 1
        good = False

        d_img = cv.imread(depth_img, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
        d_img = cv.resize(d_img, (frame.shape[1], frame.shape[0]), interpolation=cv.INTER_LINEAR)

        rendered_img = cv.imread(image, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
        rendered_img = cv.resize(rendered_img, (frame.shape[1], frame.shape[0]), interpolation=cv.INTER_LINEAR)

        if with_mask:
            d_img = np.where(last_mask, d_img, 0)
            rendered_img = np.where(last_mask, rendered_img, 0)

        d_img_scaled = np.where(d_img > depth_max, 255, 255 * (d_img / depth_max)).astype(np.uint8)
        d_img_combined = np.concatenate((frame, d_img_scaled), axis=1)
        depth_writer.write(d_img_combined)

        img = cv.resize(cv.imread(image, -1), (frame.shape[1], frame.shape[0]), interpolation=cv.INTER_LINEAR)
        squared_img_bgr, squared_depth = crop_img_opencv(frame, 256, d_img.copy())
        squared_img_rgb = cv.cvtColor(squared_img_bgr, cv.COLOR_BGR2RGB)
        real_tensor = torch.tensor(squared_img_rgb).permute([2, 0, 1]) / 256
        model.on_validation_epoch_start()
        phong_renderer = model.phong_renderer

        with torch.no_grad():
            prediction = model(imagenet_norm(real_tensor)[None])
            predicted_depth = (prediction[2][-1][0, 0] / 1e3).numpy()
            predicted_normals = prediction[-1][0].permute([1, 2, 0])
            phong = phong_renderer(
                ((prediction[2][-1][0, 0] / 1e3)[None][None], predicted_normals.permute((2, 0, 1))[None]))
            phong = phong[0].permute([1, 2, 0]).numpy()
            phong = cv.cvtColor(phong, cv.COLOR_RGB2BGR)

        phong_writer.write((phong * 255).astype(np.uint8))

        difference = squared_depth[:, :, 0] - predicted_depth

        color_mapper = MplColorHelper('bwr', -.1, .1)
        colored_difference = (color_mapper.get_rgb(difference)[:, :, :3] * 255).astype(np.uint8)

        predicted_depth = np.where(predicted_depth > depth_max, 255, 255 * (predicted_depth / depth_max)).astype(
            np.uint8)
        predicted_depth = np.stack((predicted_depth,) * 3, axis=-1)
        # predicted_normals = prediction[-1][0].permute([1, 2, 0])
        scaled_squared_depth = np.where(squared_depth > depth_max, 255, 255 * (squared_depth / depth_max)).astype(
            np.uint8)
        depth_prediction_writer.write(
            np.concatenate((scaled_squared_depth, predicted_depth, colored_difference), axis=1))
        diff_writer.write(colored_difference)
        color_prediction_write.write(np.concatenate((squared_img_bgr, predicted_depth), axis=1))
        color_writer.write(squared_img_bgr)
        depth_writer.write(scaled_squared_depth)
        prediction_writer.write(predicted_depth)
        render_writer.write(rendered_img)
        render_depth_writer.write(d_img_scaled)

        if with_mask:
            img = np.where(last_mask, img, 0)
        img_combined = np.concatenate((frame, img), axis=1)
        img_writer.write(img_combined)
    cap.release()
    img_writer.release()
    depth_writer.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    import sys

    parser = ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--directory', type=str)

    # cfg = CystoDepthConfig()
    # parser.add_arguments(CystoDepthConfig, dest='')
    args, unknown_args = parser.parse_known_args()
    # TODO: The above code fails with missing values. Need to figure out how to get it to ignore them.
    recording_directory = args.directory
    sys.argv = sys.argv[:-2]
    cysto_depth()
