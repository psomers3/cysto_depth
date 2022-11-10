import numpy as np
from PIL import Image
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.materials import Materials
from pytorch3d.renderer.cameras import PerspectiveCameras, get_ndc_to_screen_transform, get_screen_to_ndc_transform
import torch
import json
from utils.exr_utils import get_circular_mask_4_img
from typing import *
import cv2
import matplotlib.pyplot as plt


if torch.backends.mps.is_built() and torch.backends.mps.is_available():
    metal = torch.device("mps")
else:
    metal = torch.device("cuda")
torch.autograd.set_detect_anomaly(True)


def display_img(img: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show(block=True)
    # cv2.imshow('test', cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)


def get_pixel_locations(width, height):
    dim_y, dim_x = width, height
    xx_ones = torch.ones([1, 1, dim_x], dtype=torch.float, device=metal)
    yy_ones = torch.ones([1, 1, dim_y], dtype=torch.float, device=metal)

    xx_range = torch.arange(dim_y, dtype=torch.float, device=metal)
    yy_range = torch.arange(dim_x, dtype=torch.float, device=metal)
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
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        attenuation: shape is similar to input points, except last dimension is 1
    """
    light_diffuse = lights.diffuse(normals=normals, points=points)
    light_specular = lights.specular(
        normals=normals,
        points=points,
        camera_position=camera_positions,
        shininess=materials.shininess,
    )
    light_attenuation = lights.attentuation(points)
    ambient_color = materials.ambient_color * lights.ambient_color
    diffuse_color = materials.diffuse_color * light_diffuse
    specular_color = materials.specular_color * light_specular

    if normals.dim() == 2 and points.dim() == 2:
        # If given packed inputs remove batch dim in output.
        return (
            ambient_color.squeeze(),
            diffuse_color.squeeze(),
            specular_color.squeeze(),
            light_attenuation
        )

    if ambient_color.ndim != diffuse_color.ndim:
        # Reshape from (N, 3) to have dimensions compatible with
        # diffuse_color which is of shape (N, H, W, K, 3)
        ambient_color = ambient_color[:, None, None, None, :]
    return ambient_color, diffuse_color, specular_color, light_attenuation


def get_normals_from_depthmap(depth_map: torch.Tensor) -> torch.Tensor:
    zy, zx = torch.gradient(depth_map, dim=[0, 1])
    # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
    # to reduce noise
    # zx = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=5)
    # zy = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=5)

    normal = torch.dstack((-zx, -zy, torch.ones_like(depth_map)))
    # n = torch.linalg.norm(normal, axis=2)
    normal = torch.nn.functional.normalize(normal, p=2, dim=2)
    # normal[:, :, 0] /= n
    # normal[:, :, 1] /= n
    # normal[:, :, 2] /= n
    return normal


def render_rgbd(depth_map: torch.Tensor,
                color_image: torch.Tensor,
                normals_image: torch.Tensor,
                cam_intrinsic_matrix: torch.Tensor,
                spot_light,
                uniform_material) -> torch.Tensor:
    color_reshaped = color_image.reshape((color_image.shape[-3] * color_image.shape[-2], color_image.shape[-1]))
    normals_reshaped = normals_image.reshape((normals_image.shape[-3]*normals_image.shape[-2], normals_image.shape[-1]))
    pixel_locations = get_pixel_locations(depth_map.shape[1], depth_map.shape[0])
    pixel_locations = torch.cat([pixel_locations, torch.ones((*pixel_locations.shape[:-1], 1), device=metal)], dim=-1)
    rgbd_locations = pixel_locations * depth_map[None]
    flattened = rgbd_locations.reshape(1, rgbd_locations.shape[-3] * rgbd_locations.shape[-2], rgbd_locations.shape[-1])
    inv_intrinsic = torch.Tensor(torch.inverse(cam_intrinsic_matrix))
    inv_intrinsic = inv_intrinsic.to(metal)
    points_in_3d = torch.matmul(inv_intrinsic[None], torch.unsqueeze(flattened, dim=-1))
    positions = torch.squeeze(torch.squeeze(points_in_3d, dim=0), dim=-1)
    ambient_color, diffuse_color, specular_color, attenuation = apply_lighting(positions,
                                                                               normals_reshaped,
                                                                               spot_light,
                                                                               torch.Tensor([0, 0, 0])[None],
                                                                               uniform_material)
    return attenuation * ((ambient_color + diffuse_color) * color_reshaped + specular_color)


test_image = r'/Users/peter/isys/test_img/GRK011-2304.png'
test_depth = r'/Users/peter/isys/test_depth/GRK011-2304.npy'
test_normals = r'/Users/peter/isys/test_depth/GRK011-2304_normals.npy'
cam_params = json.load(open(r'test/cam_params.json'))

color = np.asarray(Image.open(test_image))
depth = np.load(test_depth)
normal = np.load(test_normals)
mask = get_circular_mask_4_img(color, .9)
masked_depth = torch.Tensor(np.expand_dims(np.where(mask, depth, 0).astype(np.float32), axis=-1))
mask = torch.Tensor(mask) > 0
mask = mask.to(metal)
mask = torch.unsqueeze(mask, dim=-1)

intrinsic = torch.Tensor(np.asarray(cam_params['IntrinsicMatrix']).T)
camera = PerspectiveCameras(focal_length=torch.Tensor([intrinsic[0, 0], intrinsic[1, 1]])[None],
                            principal_point=torch.Tensor([intrinsic[0, 2], intrinsic[1, 2]])[None],
                            image_size=torch.Tensor(color.shape)[None],
                            in_ndc=False,
                            device=metal)
normals = torch.Tensor(normal)
colors = torch.Tensor(color / 255).to(metal)
material = Materials(shininess=10, device=metal)
light = PointLights(location=((0, 0, 0),),
                    diffuse_color=((1, 1, 1),),
                    specular_color=((1, 1, 1),),
                    ambient_color=((0.1, 0.1, 0.1),),
                    attenuation_factor=(1,),
                    device=metal)

optimized_depth: torch.Tensor = masked_depth.clone()
optimized_depth = optimized_depth.to(metal)
optimized_depth.requires_grad = True
optimized_colors: torch.Tensor = torch.ones_like(colors, device=metal) * .5
optimized_colors.requires_grad = True
# optimized_normals: torch.Tensor = normals.clone()
# optimized_normals.requires_grad = True
optimizer = torch.optim.Adam([optimized_colors, optimized_depth], lr=.5)
steps = int(4e3)
loss_buffer = np.zeros(steps)
mse_loss = torch.nn.MSELoss()
cosine_loss = torch.nn.CosineSimilarity(dim=-1)
normals_from_depth = get_normals_from_depthmap(optimized_depth)
rendered_image = render_rgbd(optimized_depth,
                             optimized_colors,
                             normals_from_depth,
                             intrinsic,
                             light,
                             material).reshape(color.shape)
i = 0
loss_fig: plt.Figure = plt.figure()
ax: plt.Axes = loss_fig.add_subplot()


for i in range(steps):
    try:
        optimizer.zero_grad()
        # normals_normed = torch.nn.functional.normalize(optimized_normals, dim=-1)
        # colors_normed = torch.threshold(optimized_colors, 1, 1)
        normals_from_depth = get_normals_from_depthmap(optimized_depth)
        rendered_image = render_rgbd(optimized_depth,
                                     optimized_colors,
                                     normals_from_depth,
                                     intrinsic,
                                     light,
                                     material).reshape(color.shape)
        mse: torch.Tensor = mse_loss(torch.masked_select(rendered_image, mask), torch.masked_select(colors, mask))
        roll_up = torch.roll(normals_from_depth, 1, dims=0)
        roll_right = torch.roll(normals_from_depth, 1, dims=1)
        norm_v_loss: torch.Tensor = torch.masked_select(1 - cosine_loss(normals_from_depth, roll_up), mask)
        norm_h_loss: torch.Tensor = torch.masked_select(1 - cosine_loss(normals_from_depth, roll_right), mask)
        total_loss = .2*mse + .8*((norm_v_loss + norm_h_loss).mean()) #- 0.5*optimized_depth.mean()
        total_loss.backward()
        optimizer.step()
        loss_buffer[i] = mse.data

        ax.clear()
        ax.plot(loss_buffer[:i])
        # ax.set_ylim(0, loss_buffer[0])
        loss_fig.canvas.draw()
        loss_plot = np.frombuffer(loss_fig.canvas.tostring_rgb(), dtype='uint8')
        loss_plot = loss_plot.reshape(loss_fig.canvas.get_width_height()[::-1] + (3,))
        cv2.imshow('loss', cv2.cvtColor(loss_plot, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) == 13:
            break
    except KeyboardInterrupt:
        break


cv2.imshow('original', cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
cv2.imshow('color', cv2.cvtColor(rendered_image.cpu().detach().numpy(), cv2.COLOR_RGB2BGR))
optimized_depth = optimized_depth.cpu()
d_img = optimized_depth - torch.min(optimized_depth)
d_img /= torch.max(d_img)
d_img *= 255
cv2.imshow('depth', d_img.detach().numpy().astype(np.uint8))

# offset and rescale values to be in 0-255
normal = normals_from_depth.cpu().detach().numpy() + 1
normal /= 2
normal *= 255

cv2.imshow("normal", normal[:, :, ::-1].astype(np.uint8))
cv2.waitKey(0)
