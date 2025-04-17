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
class SlideCoordinate(BaseCoordinate):
    image_path: str
    x: int
    y: int
    kernel_size: int
    stride: int
    mpp: float
    effective_mpp: float = None
    slide_path: str = None
    slide_level: int = 0
