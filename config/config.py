from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import *
from .blender_config import BlenderConfig, ShrinkwrapConfig


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
    density: float = 4
    volume_max: float = 0.1
    scaling_range: List[float] = field(default_factory=lambda:  [0.1, 1])
    rotation_range: List[float] = field(default_factory=lambda: [0, 360])

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
