from dataclasses import dataclass

@dataclass
class BaseCoordinate:
    id: int 
    
    x: int
    y: int
    kernel_size: int
    stride: int
    
    mpp: float | None = None
    image_path: str | None = None


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


@dataclass
class PointsCoordinate:
    id: int

    x: int
    y: int
    kernel_size: int
    stride: int

    points_path: str


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

    overlap: float | None = None

