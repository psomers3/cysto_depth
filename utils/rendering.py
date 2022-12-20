import torch
from typing import *
import numpy as np
from scipy.spatial.transform import Rotation as R
from pytorch3d.renderer.lighting import PointLights as _PointLights
# keep following line, so we can import from here and make this file the only PyTorch3D direct dependency
from pytorch3d.renderer.materials import Materials


class PointLights(_PointLights):
    """ A subclass from PyTorch3D's point light to add attenuation """
    def __init__(self, *args, **kwargs):
        self.attenuation_factor = kwargs.pop('attenuation_factor')
        super(PointLights, self).__init__(*args, **kwargs)

    def attenuation(self, points) -> torch.Tensor:
        location = self.reshape_location(points)
        distance = torch.norm(location - points, dim=-1)
        return torch.unsqueeze(1 / (1 + (self.attenuation_factor*distance)), dim=-1)


def KRT_from_P(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
     Reference implementations:
      - Oxford's visual geometry group matlab toolbox
      - Scilab Image Processing toolbox
    :param P: 3x4 numpy matrix
    :return:  K, R, T such that P = K*[R | T], det(R) positive and K has positive diagonal
    """

    N = 3
    H = P[:, 0:N]  # if not numpy,  H = P.to_3x3()

    [K, R] = rf_rq(H)

    K /= K[-1, -1]

    # from http://ksimek.github.io/2012/08/14/decompose/
    # make the diagonal of K positive
    sg = np.diag(np.sign(np.diag(K)))

    K = K * sg
    R = sg * R
    # det(R) negative, just invert; the proj equation remains same:
    if np.linalg.det(R) < 0:
        R = -R
    # C = -H\P[:,-1]
    C = np.linalg.lstsq(-H, P[:, -1], rcond=1)[0]
    T = -R * C
    return K, R, T


def rf_rq(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    RQ decomposition of a numpy matrix, using only libs that already come with
    blender by default

    Author: Ricardo Fabbri
    Reference implementations:
      Oxford's visual geometry group matlab toolbox
      Scilab Image Processing toolbox
    :param P: 3x4 numpy matrix P
    :return: numpy matrices r,q
    """

    P = P.T
    # numpy only provides qr. Scipy has rq but doesn't ship with blender
    q, r = np.linalg.qr(P[::-1, ::-1], 'complete')
    q = q.T
    q = q[::-1, ::-1]
    r = r.T
    r = r[::-1, ::-1]

    if np.linalg.det(q) < 0:
        r[:, 0] *= -1
        q[0, :] *= -1
    return r, q


def get_image_size_from_intrisics(P):
    """

    :param P: 3x3 intrinsics matrix
    :return:
    """
    _P = np.zeros((3, 4))
    _P[:3, :3] = P
    P = _P
    # get krt
    K, R_world2cv, T_world2cv = KRT_from_P(np.matrix(P))
    # sensor_height_in_mm = 1  # doesn't matter
    resolution_x_in_px = K[0, 2] * 2  # principal point assumed at the center
    resolution_y_in_px = K[1, 2] * 2  # principal point assumed at the center
    return int(resolution_x_in_px), int(resolution_y_in_px)


def get_pixel_locations(width, height, device: torch.device = torch.device('cpu')):
    """
        This returns locations as channels last in image shape
    :param width:
    :param height:
    :param device:
    :return:
    """
    dim_y, dim_x = width, height
    xx_ones = torch.ones([1, 1, dim_x], dtype=torch.float, device=device)
    yy_ones = torch.ones([1, 1, dim_y], dtype=torch.float, device=device)

    xx_range = torch.arange(dim_y, dtype=torch.float, device=device)
    yy_range = torch.arange(dim_x, dtype=torch.float, device=device)
    xx_range = xx_range[None, :, None]
    yy_range = yy_range[None, :, None]

    xx_channel = torch.matmul(xx_range, xx_ones)
    yy_channel = torch.matmul(yy_range, yy_ones)

    # transpose y
    yy_channel = yy_channel.permute(0, 2, 1)
    return torch.stack([yy_channel, xx_channel], dim=-1)[0]


def phong_lighting(points, normals, lights, camera_positions, materials) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
        This expects images as channels last

    Args:
        points: torch tensor of shape (N, ..., 3) or (P, 3).
        normals: torch tensor of shape (N, ..., 3) or (P, 3)
        lights: instance of the Lights class.
        camera_positions: camera positions.
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


def get_points_in_3d(pixel_locations: torch.Tensor,
                     depth_map: torch.Tensor,
                     cam_intrinsic_matrix: torch.Tensor) -> torch.Tensor:
    pixel_locations = torch.cat([pixel_locations, torch.ones((*pixel_locations.shape[:-1], 1))], dim=-1)
    rgbd_locations = pixel_locations * depth_map[None]
    flattened = rgbd_locations.reshape(1, rgbd_locations.shape[-3] * rgbd_locations.shape[-2], rgbd_locations.shape[-1])
    inv_intrinsic = torch.Tensor(torch.inverse(cam_intrinsic_matrix))
    rotation = R.from_euler('XYZ', [0, 90, 180], degrees=True)
    # flip = torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]])  # so point cloud isn't upside down
    flip = torch.Tensor(rotation.as_matrix())  # so point cloud isn't upside down
    inv_intrinsic = flip@inv_intrinsic
    points_3d = torch.matmul(inv_intrinsic[None], torch.unsqueeze(flattened, dim=-1))
    return points_3d


def get_normals_from_3d_points(points_3d: torch.Tensor):
    dx, dy, dz = [torch.gradient(points_3d[:, :, i], dim=[0, 1], spacing=2) for i in range(3)]
    dx, dy, dz = [(d[0] + d[1])/2 for d in [dx, dy, dz]]
    gradients = torch.dstack([dx, dy, dz])
    normals = torch.nn.functional.normalize(gradients, dim=-1)
    return normals


def get_normals_from_depth_map(depth_map: torch.Tensor,
                               cam_intrinsic_matrix: torch.Tensor,
                               pixel_locations: torch.Tensor):
    points_in_3d = get_points_in_3d(pixel_locations, depth_map, cam_intrinsic_matrix)
    points_in_3d = points_in_3d.reshape((*depth_map.shape[:2], 3))
    return get_normals_from_3d_points(points_in_3d.squeeze(-1))


def render_rgbd(depth_map: torch.Tensor,
                color_image: torch.Tensor,
                normals_image: torch.Tensor,
                cam_intrinsic_matrix: torch.Tensor,
                spot_light,
                uniform_material,
                pixel_locations: torch.Tensor) -> torch.Tensor:
    color_reshaped = color_image.reshape((color_image.shape[-3] * color_image.shape[-2], color_image.shape[-1]))
    points_in_3d = get_points_in_3d(pixel_locations, depth_map, cam_intrinsic_matrix)
    positions = torch.squeeze(torch.squeeze(points_in_3d, dim=0), dim=-1)
    normals_image = normals_image.permute((1, 2, 0))
    normals_reshaped = normals_image.reshape((normals_image.shape[0] * normals_image.shape[1], 3))
    ambient_color, diffuse_color, specular_color, attenuation = phong_lighting(positions,
                                                                               normals_reshaped,
                                                                               spot_light,
                                                                               torch.Tensor([0, 0, 0])[None],
                                                                               uniform_material)
    pixels = attenuation * ((ambient_color + diffuse_color) * color_reshaped + specular_color)
    return torch.reshape(pixels, color_image.shape)
