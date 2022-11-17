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
    denoiser: str = DenoiseType.OPENIMAGEDENOISE
    device: str = 'GPU'


@dataclass
class ImageSettingsConfig:
    color_mode: str = ImageColorMode.RGB


class ThreadMode:
    AUTO = 'AUTO'
    FIXED = "FIXED"


@dataclass
class RenderConfig:
    resolution_x: int = 256
    resolution_y: int = 256
    resolution_percentage: int = 1
    image_settings: ImageSettingsConfig = ImageSettingsConfig()
    engine: str = RenderEngine.BLENDER_EEVEE
    threads_mode: str = ThreadMode.AUTO
    threads: int = 6
    use_persistent_data: bool = False
    filepath: str = os.path.join(tempfile.gettempdir(), os.getlogin())


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
    subdivision_type: str = "CATMULL_CLARK"
    render_levels: int = 3
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
    scaling_factor: float = 1
    euler_rotation: List[float] = field(default_factory=lambda: [0, 0, 0])


@dataclass
class TumorParticleConfig:
    stl_file: str = MISSING
    # density: float = 4
    volume_max: float = 0.1
    scaling_range: List[float] = field(default_factory=lambda:  [0.1, 1])
    rotation_range: List[float] = field(default_factory=lambda: [0, 360])
    amount: int = 10


@dataclass
class DiverticulumConfig:
    amount: float = 2
    subdivisions_sphere: int = 2
    radius_sphere_range: List[float] = field(default_factory=lambda: [0.001, 0.020])
    radius_opening_range: List[float] = field(default_factory=lambda: [0.0001, 0.0004])


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
    shrinkwrap: ShrinkwrapConfig = ShrinkwrapConfig()
    distance_range: List[float] = field(default_factory=lambda: [0.005, 0.05])
    # the maximum angles in degrees to randomly rotate the endoscope after positioning it.
    view_angle_max: List[float] = field(default_factory=lambda: [60, 60, 60])
    emission_range: List[float] = field(default_factory=lambda: [10, 50])
    render_normals: bool = False
    endo_image_size: List[float] = field(default_factory=lambda: [1920, 1080])
    subdivision_mod: SubdivisionModConfig = SubdivisionModConfig()
    clear_output_folder: bool = False
