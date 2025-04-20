from dataclasses import dataclass

@dataclass
class BaseCoordinate:
    x: int
    y: int
    kernel_size: int
    stride: int
    mpp: float | None = None
    image_path: str | None = None


@dataclass
class SlideCoordinate:
    x: int
    y: int
    kernel_size: int
    stride: int
    mpp: float
    image_path: str

    target_mpp: float
    effective_mpp: float
    scale_factor: float
    overlap: float | None = None

