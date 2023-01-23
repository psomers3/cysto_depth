from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import *
import os
import tempfile


class RenderEngine:
    BLENDER_EEVEE: str = 'BLENDER_EEVEE'
    CYCLES: str = 'CYCLES'


class ImageColorMode:
    RGB: str = 'RGB'


class DenoiseType:
    OPENIMAGEDENOISE: str = 'OPENIMAGEDENOISE'
    OPTIX: str = "OPTIX"


@dataclass
class CyclesConfig:
    adaptive_min_samples: int = 64
    adaptive_max_samples: int = 128
    samples: int = 64
    denoiser: str = DenoiseType.OPENIMAGEDENOISE
    denoising_prefilter: str = 'FAST'
    device: str = 'GPU'
    device_type: str = 'OPTIX'
    use_auto_tile: bool = False


@dataclass
class ImageSettingsConfig:
    color_mode: str = ImageColorMode.RGB


class ThreadMode:
    AUTO = 'AUTO'
    FIXED = "FIXED"


def get_login():
    try:
        login = os.getlogin()
    except OSError:
        return None
    return login


@dataclass
class RenderConfig:
    resolution_x: int = 256
    resolution_y: int = 256
    resolution_percentage: int = 100
    image_settings: ImageSettingsConfig = ImageSettingsConfig()
    engine: str = RenderEngine.BLENDER_EEVEE
    threads_mode: str = ThreadMode.AUTO
    threads: int = 6
    use_persistent_data: bool = False
    filepath: str = field(default_factory=lambda:
                          tempfile.gettempdir() if not get_login() else
                          os.path.join(tempfile.gettempdir(), get_login()))


class LengthUnits:
    MILLIMETERS = "MILLIMETERS"
    METERS = "METERS"


@dataclass
class UnitSettingsConfig:
    length_unit: str = LengthUnits.MILLIMETERS


@dataclass
class ShrinkwrapConfig:
    wrap_mode: str = "ON_SURFACE"
    distance: float = .005
    use_track_normal: bool = False
    track_axis: str = "TRACK_NEGATIVE_Z"
    shrinkwrap_type: str = "PROJECT"
    project_axis: str = "NEG_Z"


@dataclass
class SubdivisionModConfig:
    uv_smooth: str = "PRESERVE_BOUNDARIES"
    subdivision_type: str = "SIMPLE"
    render_levels: int = 2
    levels: int = 2


@dataclass
class BlenderConfig:
    render: RenderConfig = RenderConfig()

    cycles: CyclesConfig = CyclesConfig()
    """The options for the cycles engine. Only matters if render engine is "CYCLES"""

    unit_settings: UnitSettingsConfig = UnitSettingsConfig()
    use_nodes: bool = True


@dataclass
class EndoLightConfig:
    stl_file: str = MISSING
    emission_color: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    emission_strength: int = 50
    """ surface light strength in Watts for entire surface area """
    scaling_factor: float = 1
    euler_rotation: List[float] = field(default_factory=lambda: [0, 0, 0])


@dataclass
class TumorParticleConfig:
    stl_file: str = MISSING
    volume_max: float = 0.1
    scaling_range: List[float] = field(default_factory=lambda: [0.1, 1])
    rotation_mode: str = 'random'
    """ Determines how the particle instances are rotated (default: 'random')
            - 'random': random rotation according to the given rotation_range 
            - 'align_to_surface': z-axis of particle model is aligned to the normal of the surface"""
    rotation_range: List[float] = field(default_factory=lambda: [0, 360])
    amount: int = 10


@dataclass
class DiverticulumConfig:
    amount: float = 2
    subdivisions_sphere: int = 2
    radius_sphere_range: List[float] = field(default_factory=lambda: [0.001, 0.020])
    translation_range: List[float] = field(default_factory=lambda: [0.0001, 0.0004])


@dataclass
class ResectionLoopConfig:
    wire_stl: str = MISSING
    """ path to the STL for the wire """
    insulation_stl: str = MISSING
    """ path to the STL for the insulation """
    extension_direction: List[float] = field(default_factory=lambda: [1, 0, 0])
    """ direction in which the tool is extended in the coordinates of the stl-file"""
    no_clip_points: List[List[float]] = field(default_factory=lambda: [[0, 0, 0]])
    """ points of the tool where clipping is prevented in the coordinates of the stl-file"""
    scaling_factor: float = 0.001
    """ initial scaling """
    euler_rotation: List[float] = field(default_factory=lambda: [90, 0, 0])
    """ initial rotation on load (euler XYZ)"""
    max_extension: float = 10
    """ maximum distance the wire can be extended from the endoscope """
    extension_direction: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    """ direction in which the tool is extended in the coordinates of the stl-file"""
    no_clip_points: List[List[float]] = field(default_factory=lambda: [[0.0, 0.0, 0.0]])
    """ points of the tool where clipping is prevented in the coordinates of the stl-file"""
    max_retraction: float = 5
    """ maximum distance the insulation can be retracted from the wire """
    wire_base_color: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    """ material of wire: base color """
    wire_metallic: float = 1
    """ material of wire: value of the shader property 'Metallic' """
    wire_roughness: float = 0.2
    """ material of wire: roughness """
    wire_anisotropic: float = 1
    """ material of wire: value of the shader property 'Anisotropic' """

@dataclass
class BladderMaterialConfig:
    volume_scatter_density: float = 10.0
    volume_scatter_anisotropy: float = 0.0
    volume_absorbtion_density: float = 10.0


@dataclass
class BladderMaterialConfig:
    volume_scatter_density: float = 30.0
    volume_scatter_anisotropy: float = 0.9
    """ neg. values scatter back toward the light source, positive forward """
    volume_absorbtion_density: float = 5.0


@dataclass
class MainConfig:
    blender: BlenderConfig = BlenderConfig()
    # path to where the bladder STL models are. This will be searched recursively
    models_dir: str = MISSING
    # the regex term to use for finding bladder models
    bladder_model_regex: str = 'smooth'
    # file storing the 3x3 camera intrinsics matrix
    camera_intrinsics: str = MISSING
    bladder_volume: float = 400
    endo_light: EndoLightConfig = EndoLightConfig()
    tumor_particles: TumorParticleConfig = TumorParticleConfig()
    diverticulum: DiverticulumConfig = DiverticulumConfig()
    output_folder: str = MISSING
    samples_per_model: int = 3
    shrinkwrap_wire: ShrinkwrapConfig = ShrinkwrapConfig()
    shrinkwrap_tool: ShrinkwrapConfig = ShrinkwrapConfig()
    distance_range: List[float] = field(default_factory=lambda: [0.005, 0.05])
    # the maximum angles in degrees to randomly rotate the endoscope after positioning it.
    view_angle_max: List[float] = field(default_factory=lambda: [60, 60, 60])
    emission_range: List[float] = field(default_factory=lambda: [10, 50])
    render_normals: bool = False
    endo_image_size: List[float] = field(default_factory=lambda: [1920, 1080])
    subdivision_mod: SubdivisionModConfig = SubdivisionModConfig()
    clear_output_folder: bool = False
    resection_loop: ResectionLoopConfig = ResectionLoopConfig()
    endoscope_angle: float = 30
    materials_files: List[str] = field(default_factory=lambda: [])
    """ list of .blend files containing materials that will be imported """
    bladder_materials: List[str] = field(default_factory=lambda: [])
    """ list of different bladder wall materials to use during rendering """
    bladder_material_config: BladderMaterialConfig = BladderMaterialConfig()
    """ Control over parameters inside the bladder material (i.e. volume scattering) """