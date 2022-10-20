from dataclasses import dataclass


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
    resolution_percentage: float = 1
    image_settings: ImageSettingsConfig = ImageSettingsConfig()
    engine: str = RenderEngine.BLENDER_EEVEE
    threads_mode: str = ThreadMode.AUTO
    threads: int = 6


class LengthUnits:
    MILLIMETERS = "MILLIMETERS"
    METERS = "METERS"


@dataclass
class UnitSettingsConfig:
    length_unit = LengthUnits.MILLIMETERS


@dataclass
class ShrinkwrapConfig:
    wrap_mode: str = "ON_SURFACE"
    distance: float = .005
    use_track_normal: bool = False
    track_axis: str = "TRACK_NEGATIVE_Z"
    shrinkwrap_type: str = "PROJECT"
    project_axis: str = "POS_Z"


@dataclass
class BlenderConfig:
    render: RenderConfig = RenderConfig()

    cycles: CyclesConfig = CyclesConfig()
    """The options for the cycles engine. Only matters if render engine is "CYCLES"""

    unit_settings: UnitSettingsConfig = UnitSettingsConfig()
