from dataclasses import dataclass, field
from uuid import uuid4, UUID

@dataclass
class BaseCoordinate:
    id: int 
    
    x: int
    y: int
    kernel_size: int
    stride: int
    
    mpp: float | None = None
    image_path: str | None = None
    uuid: str = field(default_factory=lambda: str(uuid4()))

@dataclass
class Coordinate:

    id: int
    x: int
    y: int
    kernel_size: int
    stride: int

    image_path: str | None = None
    sample_id: str | None = None
    uuid: str = field(default_factory=lambda: str(uuid4()))

@dataclass
class RandomCoordinate:

    id: int
    x: int
    y: int
    kernel_size: int

    image_path: str | None = None
    sample_id: str | None = None
    uuid: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class SlideCoordinate:
    id: int

    x: int
    y: int
    kernel_size: int
    stride: int
    mpp: float

    image_path: str

    patch_size: int
    patch_stride: int
    patch_mpp: float
    target_mpp: float
    scale_factor: float

    overlap: float = -1
    uuid: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class PointsCoordinate:
    id: int

    x: int
    y: int
    kernel_size: int
    stride: int

    points_path: str
    uuid: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class XeniumCoordinate:
    id: int

    x: int
    y: int
    kernel_size: int
    stride: int
    mpp: float

    image_path: str
    points_path: str

    patch_size: int
    patch_stride: int
    patch_mpp: float
    target_mpp: float
    scale_factor: float

    global_id: str
    sample_id: str

    overlap: float | None = None
    uuid: str = field(default_factory=lambda: str(uuid4()))

