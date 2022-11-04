import numpy as np
from PIL import Image
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.materials import Materials
from pytorch3d.renderer.cameras import PerspectiveCameras, get_ndc_to_screen_transform, get_screen_to_ndc_transform
import torch
import json
from utils.exr_utils import get_circular_mask_4_img
from typing import *


def get_pixel_locations(width, height):
    dim_y, dim_x = width, height
    xx_ones = torch.ones([1, 1, dim_x], dtype=torch.int32)
    yy_ones = torch.ones([1, 1, dim_y], dtype=torch.int32)

    xx_range = torch.arange(dim_y, dtype=torch.int32)
    yy_range = torch.arange(dim_x, dtype=torch.int32)
    xx_range = xx_range[None, :, None]
    yy_range = yy_range[None, :, None]

    xx_channel = torch.matmul(xx_range, xx_ones)
    yy_channel = torch.matmul(yy_range, yy_ones)

    # transpose y
    yy_channel = yy_channel.permute(0, 2, 1)

    # xx_channel = xx_channel.float() / (dim_y - 1)
    # yy_channel = yy_channel.float() / (dim_x - 1)
    #
    # xx_channel = xx_channel * 2 - 1
    # yy_channel = yy_channel * 2 - 1
    return torch.stack([yy_channel, xx_channel], dim=-1)


def apply_lighting(points, normals, lights, camera_positions, materials) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        points: torch tensor of shape (N, ..., 3) or (P, 3).
        normals: torch tensor of shape (N, ..., 3) or (P, 3)
        lights: instance of the Lights class.
        cameras: instance of the Cameras class.
        materials: instance of the Materials class.

    Returns:
        ambient_color: same shape as materials.ambient_color
        diffuse_color: same shape as the input points
        specular_color: same shape as the input points
    """
    light_diffuse = lights.diffuse(normals=normals, points=points)
    light_specular = lights.specular(
        normals=normals,
        points=points,
        camera_position=camera_positions,
        shininess=materials.shininess,
    )
    ambient_color = materials.ambient_color * lights.ambient_color
    diffuse_color = materials.diffuse_color * light_diffuse
    specular_color = materials.specular_color * light_specular

    if normals.dim() == 2 and points.dim() == 2:
        # If given packed inputs remove batch dim in output.
        return (
            ambient_color.squeeze(),
            diffuse_color.squeeze(),
            specular_color.squeeze(),
        )

    if ambient_color.ndim != diffuse_color.ndim:
        # Reshape from (N, 3) to have dimensions compatible with
        # diffuse_color which is of shape (N, H, W, K, 3)
        ambient_color = ambient_color[:, None, None, None, :]
    return ambient_color, diffuse_color, specular_color


test_image = r'/Users/peter/isys/test_img/GRK011-2304.png'
test_depth = r'/Users/peter/isys/test_depth/GRK011-2304.npy'
test_normals = r'/Users/peter/isys/test_depth/GRK011-2304_normals.npy'
cam_params = json.load(open(r'test/cam_params.json'))

color = np.asarray(Image.open(test_image))
depth = np.load(test_depth)
normal = np.load(test_normals)
mask = get_circular_mask_4_img(color, .9)
masked_depth = torch.Tensor(np.expand_dims(np.where(mask, depth, 1e5).astype(np.float32), axis=-1))

intrinsic = torch.Tensor(np.asarray(cam_params['IntrinsicMatrix']).T)
inv_intrinsic = torch.Tensor(np.linalg.inv(intrinsic))

camera = PerspectiveCameras(focal_length=torch.Tensor([intrinsic[0, 0], intrinsic[1, 1]])[None],
                            principal_point=torch.Tensor([intrinsic[0, 2], intrinsic[1, 2]])[None],
                            image_size=torch.Tensor(color.shape)[None],
                            in_ndc=False)
pixel_locations = get_pixel_locations(color.shape[1], color.shape[0])
pixel_locations = torch.cat([pixel_locations, torch.ones((*pixel_locations.shape[:-1], 1))], dim=-1)

rgbd_locations = pixel_locations*masked_depth[None]
flattened = rgbd_locations.reshape(1, rgbd_locations.shape[-3]*rgbd_locations.shape[-2], rgbd_locations.shape[-1])
points_in_3d = torch.matmul(inv_intrinsic[None], torch.unsqueeze(flattened, dim=-1))
positions = torch.squeeze(torch.squeeze(points_in_3d, dim=0), dim=-1)
normals = torch.Tensor(normal.reshape((normal.shape[-3]*normal.shape[-2], normal.shape[-1])))
colors = torch.Tensor(color.reshape((color.shape[-3]*color.shape[-2], color.shape[-1])).copy()) / 255
material = Materials(shininess=.2)
light = PointLights(location=((0, 0, 0),),
                    diffuse_color=((1, 1, 1),),
                    specular_color=((.2, .2, .2),),
                    ambient_color=((0.2, 0.2, 0.2),))

ambient_color, diffuse_color, specular_color = apply_lighting(positions,
                                                              normals,
                                                              light,
                                                              torch.Tensor([0, 0, 0])[None],
                                                              material)
colors = (ambient_color + diffuse_color) * colors + specular_color
rendered_image = colors.detach().numpy().reshape(color.shape)
import cv2
cv2.imshow('test', cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)






# intrinsic_cam = o3d.camera.PinholeCameraIntrinsic()
# intrinsic_cam.intrinsic_matrix = intrinsic_matrix
# intrinsic_cam.width = color.shape[1]
# intrinsic_cam.height = color.shape[0]
# op3_cam = o3d.camera.PinholeCameraParameters()
# op3_cam.intrinsic = intrinsic_cam
#
# rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color=o3d.geometry.Image(color),
#                                                           depth=o3d.geometry.Image(masked_depth),
#                                                           depth_trunc=1e4,
#                                                           depth_scale=1,
#                                                           convert_rgb_to_intensity=False)
# points: o3d.geometry.PointCloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, op3_cam.intrinsic)
