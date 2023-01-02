import torch
from typing import *
import numpy as np
from scipy.spatial.transform import Rotation as R
from pytorch3d.renderer.lighting import diffuse, _validate_light_properties, TensorProperties
from pytorch3d.renderer.lighting import convert_to_tensors_and_broadcast, F
import pytorch3d.renderer.lighting as pytorch3d_lighting

# keep following line, so we can import from here and make this file the only PyTorch3D direct dependency
from pytorch3d.renderer.materials import Materials


def specular(
        points, normals, direction, color, camera_position, shininess
) -> torch.Tensor:
    """
    Calculate the specular component of light reflection using Blinn-Phong Shading.

    Args:
        points: (N, ..., 3) xyz coordinates of the points.
        normals: (N, ..., 3) xyz normal vectors for each point.
        color: (N, 3) RGB color of the specular component of the light.
        direction: (N, 3) vector direction of the light.
        camera_position: (N, 3) The xyz position of the camera.
        shininess: (N)  The specular exponent of the material.

    Returns:
        colors: (N, ..., 3), same shape as the input points.

    The points, normals, camera_position, and direction should be in the same
    coordinate frame i.e. if the points have been transformed from
    world -> view space then the normals, camera_position, and light direction
    should also be in view space.

    To use with a batch of packed points reindex in the following way.
    .. code-block:: python::

        Args:
            points: (P, 3)
            normals: (P, 3)
            color: (N, 3)[batch_idx] -> (P, 3)
            direction: (N, 3)[batch_idx] -> (P, 3)
            camera_position: (N, 3)[batch_idx] -> (P, 3)
            shininess: (N)[batch_idx] -> (P)
        Returns:
            colors: (P, 3)

        where batch_idx is of shape (P). For meshes batch_idx can be:
        meshes.verts_packed_to_mesh_idx() or meshes.faces_packed_to_mesh_idx().
    """
    # TODO: handle multiple directional lights
    # TODO: attenuate based on inverse squared distance to the light source

    if points.shape != normals.shape:
        msg = "Expected points and normals to have the same shape: got %r, %r"
        raise ValueError(msg % (points.shape, normals.shape))

    # Ensure all inputs have same batch dimension as points
    matched_tensors = convert_to_tensors_and_broadcast(
        points, color, direction, camera_position, shininess, device=points.device
    )
    _, color, direction, camera_position, shininess = matched_tensors

    # Reshape direction and color so they have all the arbitrary intermediate
    # dimensions as points. Assume first dim = batch dim and last dim = 3.
    points_dims = points.shape[1:-1]
    expand_dims = (-1,) + (1,) * len(points_dims)
    if direction.shape != normals.shape:
        direction = direction.view(expand_dims + (3,))
    if color.shape != normals.shape:
        color = color.view(expand_dims + (3,))
    if camera_position.shape != normals.shape:
        camera_position = camera_position.view(expand_dims + (3,))
    if shininess.shape != normals.shape:
        shininess = shininess.view(expand_dims)

    # Renormalize the normals in case they have been interpolated.
    # We tried a version that uses F.cosine_similarity instead of renormalizing,
    # but it was slower.
    normals = F.normalize(normals, p=2, dim=-1, eps=1e-6)
    direction = F.normalize(direction, p=2, dim=-1, eps=1e-6)

    # Calculate the specular reflection.
    view_direction = camera_position - points
    view_direction = F.normalize(view_direction, p=2, dim=-1, eps=1e-6)
    halfway_direction = F.normalize(view_direction + direction, p=2, dim=1, eps=1e-6)
    cos_angle = torch.sum(normals * halfway_direction, dim=-1)
    # No specular highlights if angle is less than 0.
    mask = (cos_angle > 0).to(torch.float32)

    # reflect_direction = -direction + 2 * (cos_angle[..., None] * normals)

    # Cosine of the angle between the reflected light ray and the viewer
    alpha = cos_angle * mask
    return color * torch.pow(alpha, shininess)[..., None]


class PointLights(TensorProperties):
    def __init__(
        self,
        ambient_color=((0.5, 0.5, 0.5),),
        diffuse_color=((0.3, 0.3, 0.3),),
        specular_color=((0.2, 0.2, 0.2),),
        location=((0, 1, 0),),
        attenuation_factor=((4),),
        device: str = "cpu",
    ) -> None:
        """
        Args:
            ambient_color: RGB color of the ambient component
            diffuse_color: RGB color of the diffuse component
            specular_color: RGB color of the specular component
            location: xyz position of the light.
            device: Device (as str or torch.device) on which the tensors should be located

        The inputs can each be
            - 3 element tuple/list or list of lists
            - torch tensor of shape (1, 3)
            - torch tensor of shape (N, 3)
        The inputs are broadcast against each other so they all have batch
        dimension N.
        """
        super().__init__(
            device=device,
            ambient_color=ambient_color,
            diffuse_color=diffuse_color,
            specular_color=specular_color,
            location=location,
            attenuation_factor=attenuation_factor,
        )
        _validate_light_properties(self)
        if self.location.shape[-1] != 3:
            msg = "Expected location to have shape (N, 3); got %r"
            raise ValueError(msg % repr(self.location.shape))

    def clone(self):
        other = self.__class__(device=self.device)
        return super().clone(other)

    def reshape_location(self, points) -> torch.Tensor:
        """
        Reshape the location tensor to have dimensions
        compatible with the points which can either be of
        shape (P, 3) or (N, H, W, K, 3).
        """
        if self.location.ndim == points.ndim:
            # pyre-fixme[7]
            return self.location
        # pyre-fixme[29]
        return self.location[:, None, None, None, :]

    def diffuse(self, normals, points) -> torch.Tensor:
        location = self.reshape_location(points)
        direction = location - points
        return diffuse(normals=normals, color=self.diffuse_color, direction=direction)

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        location = self.reshape_location(points)
        direction = location - points
        return specular(
            points=points,
            normals=normals,
            color=self.specular_color,
            direction=direction,
            camera_position=camera_position,
            shininess=shininess,
        )

    def attenuation(self, points) -> torch.Tensor:
        location = self.reshape_location(points)
        distance = torch.norm(location - points, dim=-1)
        attenuation = torch.clamp(torch.unsqueeze((1 / (1 + (self.attenuation_factor * distance))), dim=-1), 0, 1)
        return attenuation


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
    light_attenuation = lights.attenuation(points)
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
        ambient_color = ambient_color[None]
    return ambient_color, diffuse_color, specular_color, light_attenuation


def get_points_in_3d(pixel_locations: torch.Tensor,
                     depth_map: torch.Tensor,
                     cam_intrinsic_matrix: torch.Tensor) -> torch.Tensor:
    if len(depth_map.shape) < 4:
        depth_map = depth_map[None]
    pixel_locations = torch.cat([pixel_locations, torch.ones((*pixel_locations.shape[:-1], 1))], dim=-1)
    rgbd_locations = pixel_locations * depth_map
    flattened = rgbd_locations.reshape(depth_map.shape[0], rgbd_locations.shape[-3] * rgbd_locations.shape[-2],
                                       rgbd_locations.shape[-1])
    inv_intrinsic = torch.Tensor(torch.inverse(cam_intrinsic_matrix))
    rotation = R.from_euler('XYZ', [0, 90, 180], degrees=True)  # obtained with guess and check... may not be correct
    flip = torch.Tensor(rotation.as_matrix())  # so point cloud isn't upside down
    inv_intrinsic = flip @ inv_intrinsic
    points_3d = torch.matmul(inv_intrinsic[None], torch.unsqueeze(flattened, dim=-1))
    return points_3d


def get_normals_from_3d_points(points_3d: torch.Tensor):
    dx, dy, dz = [torch.gradient(points_3d[:, :, i], dim=[0, 1], spacing=2) for i in range(3)]
    dx, dy, dz = [(d[0] + d[1]) / 2 for d in [dx, dy, dz]]
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
    """
    expected channels last
    :param depth_map:
    :param color_image:
    :param normals_image:
    :param cam_intrinsic_matrix:
    :param spot_light:
    :param uniform_material:
    :param pixel_locations:
    :return:
    """
    had_batch_dim = True
    if len(depth_map.shape) < 4:
        had_batch_dim = False
        depth_map = depth_map[None]
    if len(color_image.shape) < 4:
        color_image = color_image[None]
    if len(normals_image.shape) < 4:
        normals_image = normals_image[None]
    batch_size = depth_map.shape[0]
    color_reshaped = color_image.reshape((color_image.shape[0], color_image.shape[1] * color_image.shape[2], 3))
    points_in_3d = get_points_in_3d(pixel_locations, depth_map, cam_intrinsic_matrix)
    positions = torch.squeeze(torch.squeeze(points_in_3d, dim=1), dim=-1)
    normals_reshaped = normals_image.reshape(
        (normals_image.shape[0], normals_image.shape[1] * normals_image.shape[2], 3))

    camera_positions = torch.Tensor((((0, 0, 0),),))
    if batch_size == 1:
        color_image = color_image.squeeze(0)
        normals_reshaped = normals_reshaped.squeeze(0)
        positions = positions.squeeze(0)
        camera_positions = camera_positions.squeeze(0)

    ambient_color, diffuse_color, specular_color, attenuation = phong_lighting(positions,
                                                                               normals_reshaped,
                                                                               spot_light,
                                                                               camera_positions,
                                                                               uniform_material)
    pixels = attenuation * ((ambient_color + diffuse_color + specular_color) * color_reshaped)
    if had_batch_dim:
        if batch_size == 1:
            return pixels.reshape(color_image.shape)[None]
        else:
            return pixels.reshape(normals_image.shape)
    else:
        return pixels.reshape(color_image.shape)
