import kornia.geometry.depth
import torch
from typing import *
import numpy as np
from pytorch3d.renderer.lighting import diffuse, _validate_light_properties, TensorProperties
from pytorch3d.renderer.lighting import convert_to_tensors_and_broadcast, F
from kornia.core import Tensor
from kornia.utils import create_meshgrid
from kornia.geometry.camera import unproject_points
from kornia.filters import spatial_gradient
from data.data_transforms import Squarify
from config.training_config import PhongConfig
from scipy.spatial.transform import Rotation
# keep following line, so we can import from here and make this file the only PyTorch3D direct dependency
from pytorch3d.renderer.materials import Materials


def blinn_specular(
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
    halfway_direction = F.normalize(view_direction + direction, p=2, dim=-1, eps=1e-6)
    cos_angle = torch.sum(normals * halfway_direction, dim=-1)
    # Cosine of the angle between the reflected light ray and the viewer
    alpha = F.relu(cos_angle)
    return color * torch.pow(alpha, shininess)[..., None]


class PointLights(TensorProperties):
    def __init__(
        self,
        ambient_color=((0.5, 0.5, 0.5),),
        diffuse_color=((0.3, 0.3, 0.3),),
        specular_color=((0.2, 0.2, 0.2),),
        location=((0, 0, 0),),
        attenuation_factor=((4),),
        device: torch.device = "cpu",
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

        if self.location.shape[0] != points.shape[0]:
            return self.location.repeat_interleave(points.shape[0], dim=0)[:, None, None, :]
        # pyre-fixme[29]
        return self.location[:, None, None, :]

    def diffuse(self, normals, points) -> torch.Tensor:
        location = self.reshape_location(points)
        direction = location - points
        return diffuse(normals=normals, color=self.diffuse_color, direction=direction)

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        location = self.reshape_location(points)
        direction = location - points
        return blinn_specular(
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


def depth_to_3d(depth: Tensor, camera_matrix: Tensor, pixel_grid: torch.Tensor = None, normalize_points: bool = False) -> Tensor:
    """Compute a 3d point per pixel given its depth value and the camera intrinsics.

    Args:
        depth: image tensor containing a depth value per pixel with shape :math:`(B, 1, H, W)`.
        camera_matrix: tensor containing the camera intrinsics with shape :math:`(B, 3, 3)`.
        pixel_grid: tensor containg pixel locations corresponding to depth with shape :math:`(B, H, W, 2)`
        normalize_points: whether to normalize the pointcloud. This must be set to `True` when the depth is
          represented as the Euclidean ray length from the camera position.

    Return:
        tensor with a 3d point per pixel of the same resolution as the input :math:`(B, 3, H, W)`.

    Example:
        >>> depth = torch.rand(1, 1, 4, 4)
        >>> K = torch.eye(3)[None]
        >>> depth_to_3d(depth, K).shape
        torch.Size([1, 3, 4, 4])
    """
    if not isinstance(depth, Tensor):
        raise TypeError(f"Input depht type is not a Tensor. Got {type(depth)}.")

    if not (len(depth.shape) == 4 and depth.shape[-3] == 1):
        raise ValueError(f"Input depth musth have a shape (B, 1, H, W). Got: {depth.shape}")

    if not isinstance(camera_matrix, Tensor):
        raise TypeError(f"Input camera_matrix type is not a Tensor. " f"Got {type(camera_matrix)}.")

    if not (len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3)):
        raise ValueError(f"Input camera_matrix must have a shape (B, 3, 3). " f"Got: {camera_matrix.shape}.")

    # create base coordinates grid
    _, _, height, width = depth.shape
    if pixel_grid is not None:
        points_2d: Tensor = pixel_grid
    else:
        points_2d: Tensor = create_meshgrid(height, width, normalized_coordinates=False)  # 1xHxWx2
    points_2d = points_2d.to(depth.device).to(depth.dtype)

    # depth should come in Bx1xHxW
    points_depth: Tensor = depth.permute(0, 2, 3, 1)  # 1xHxWx1

    # project pixels to camera frame
    camera_matrix_tmp: Tensor = camera_matrix[:, None, None]  # Bx1x1x3x3
    points_3d: Tensor = unproject_points(
        points_2d, points_depth, camera_matrix_tmp, normalize=normalize_points
    )  # BxHxWx3

    return points_3d.permute(0, 3, 1, 2)  # Bx3xHxW


kornia.geometry.depth.depth_to_3d = depth_to_3d


def depth_to_normals(depth: Tensor, camera_matrix: Tensor, pixel_grid: torch.Tensor = None, normalize_points: bool = False) -> Tensor:
    """Compute the normal surface per pixel.

    Args:
        depth: image tensor containing a depth value per pixel with shape :math:`(B, 1, H, W)`.
        camera_matrix: tensor containing the camera intrinsics with shape :math:`(B, 3, 3)`.
        pixel_grid: tensor containg pixel locations corresponding to depth with shape :math:`(B, H, W, 2)`
        normalize_points: whether to normalize the pointcloud. This must be set to `True` when the depth is
        represented as the Euclidean ray length from the camera position.

    Return:
        tensor with a normal surface vector per pixel of the same resolution as the input :math:`(B, 3, H, W)`.

    Example:
        >>> depth = torch.rand(1, 1, 4, 4)
        >>> K = torch.eye(3)[None]
        >>> depth_to_normals(depth, K).shape
        torch.Size([1, 3, 4, 4])
    """
    if not isinstance(depth, Tensor):
        raise TypeError(f"Input depht type is not a Tensor. Got {type(depth)}.")

    if not (len(depth.shape) == 4 and depth.shape[-3] == 1):
        raise ValueError(f"Input depth musth have a shape (B, 1, H, W). Got: {depth.shape}")

    if not isinstance(camera_matrix, Tensor):
        raise TypeError(f"Input camera_matrix type is not a Tensor. " f"Got {type(camera_matrix)}.")

    if not (len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3)):
        raise ValueError(f"Input camera_matrix must have a shape (B, 3, 3). " f"Got: {camera_matrix.shape}.")

    # compute the 3d points from depth
    xyz: Tensor = depth_to_3d(depth, camera_matrix, pixel_grid=pixel_grid, normalize_points=normalize_points)  # Bx3xHxW

    # compute the pointcloud spatial gradients
    gradients: Tensor = spatial_gradient(xyz)  # Bx3x2xHxW

    # compute normals
    a, b = gradients[:, :, 0], gradients[:, :, 1]  # Bx3xHxW

    normals: Tensor = torch.cross(a, b, dim=1)  # Bx3xHxW
    return F.normalize(normals, dim=1, p=2)


kornia.geometry.depth_to_normals = depth_to_normals


def render_rgbd(depth_map: torch.Tensor,
                color_image: torch.Tensor,
                normals_image: torch.Tensor,
                cam_intrinsic_matrix: torch.Tensor,
                spot_light,
                uniform_material,
                pixel_locations: torch.Tensor,
                device: torch.device = None) -> torch.Tensor:
    """
    expected channels last
    :param depth_map:
    :param color_image:
    :param normals_image:
    :param cam_intrinsic_matrix:
    :param spot_light:
    :param uniform_material:
    :param pixel_locations:
    :param device:
    :return:
    """
    had_batch_dim = True
    if depth_map.ndim < 4:
        had_batch_dim = False
        depth_map = depth_map[None]
    if color_image.ndim < 4:
        color_image = color_image[None]
    if normals_image.ndim < 4:
        normals_image = normals_image[None]
    if cam_intrinsic_matrix.ndim < 4:
        cam_intrinsic_matrix = cam_intrinsic_matrix[None]

    points_in_3d = depth_to_3d(depth_map, camera_matrix=cam_intrinsic_matrix, pixel_grid=pixel_locations)
    positions = torch.squeeze(torch.squeeze(points_in_3d, dim=1), dim=-1)
    positions = positions.permute([0, 2, 3, 1])
    camera_positions = torch.Tensor((((0, 0, 0),),)).to(device)
    ambient_color, diffuse_color, specular_color, attenuation = phong_lighting(positions,
                                                                               -normals_image,
                                                                               spot_light,
                                                                               camera_positions,
                                                                               uniform_material)
    pixels = attenuation * ((ambient_color + diffuse_color + specular_color) * color_image)
    if had_batch_dim:
        return pixels
    else:
        return pixels.squeeze(0)


class PhongRender(torch.nn.Module):
    def __init__(self, config: PhongConfig, image_size: int = 256, device=None) -> None:
        super(PhongRender, self).__init__()
        self.config = config
        self.camera_intrinsics = torch.Tensor(config.camera_intrinsics, device='cpu')
        self.camera_intrinsics.requires_grad_(False)
        self.squarify = Squarify(image_size)
        # get the original camera pixel locations at the desired image resolution
        original_image_size = get_image_size_from_intrisics(self.camera_intrinsics)
        self.camera_intrinsics = self.camera_intrinsics.to(device)
        pixels = get_pixel_locations(*original_image_size)
        self.resized_pixel_locations = self.squarify(torch.permute(pixels, (2, 0, 1))).to(device)
        self.resized_pixel_locations = torch.permute(self.resized_pixel_locations, (1, 2, 0)) - self.camera_intrinsics[
            -1, [1, 0]]
        self.resized_pixel_locations = self.resized_pixel_locations.to(device)
        self.resized_pixel_locations.requires_grad_(False)
        self.grey = torch.ones((image_size, image_size, 3), device=device) * .5
        self.grey.requires_grad_(False)
        self.material = Materials(shininess=config.material_shininess, device=device)
        self.material.requires_grad_(False)
        self.light = PointLights(location=((0, 0, 0),),
                                 diffuse_color=(config.diffusion_color,),
                                 specular_color=(config.specular_color,),
                                 ambient_color=(config.ambient_color,),
                                 attenuation_factor=config.attenuation,
                                 device=device)
        self.light.requires_grad_(False)
        self.device = device

    def forward(self, predicted_depth_normals: Tuple[torch.Tensor, ...]) \
            -> torch.Tensor:
        """

        :param predicted_depth_normals: tuple of (depth, normals)
        :type predicted_depth_normals: Tuple[torch.Tensor, ...]
        :return: the loss value and the rendered images
        """
        depth, normals = predicted_depth_normals
        rendered = render_rgbd(depth,
                               self.grey,
                               normals.permute((0, 2, 3, 1)),
                               self.camera_intrinsics,
                               self.light,
                               self.material,
                               self.resized_pixel_locations,
                               device=self.device)
        rendered = rendered.permute(0, 3, 1, 2)
        return rendered
    
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """

        :param predicted_depth_normals: tuple of (depth, normals)
        :type predicted_depth_normals: Tuple[torch.Tensor, ...]
        :return: the loss value and the rendered images
        """
        return super(PhongRender, self).__call__(*args, **kwargs)
