from uuid import uuid4
from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path

class BaseCoordinate(BaseModel):
    """Core coordinate fields shared by all coordinate types.

    A unique identifier is assigned on creation using ``uuid4``.
    """
    model_config = ConfigDict(extra='allow')

    x: int
    y: int
    uuid: str = Field(default_factory=lambda: str(uuid4()))


class PatchCoordinate(BaseCoordinate):
    """Base class adding a ``stride`` field to ``BasicCoordinate``."""

    kernel_size: int
    image_path: str
    stride: int | None = None


class SlideCoordinate(BaseCoordinate):
    """Coordinate for slide-based extraction with patch parameters.

    It has a mandatory ``mpp`` (microns per pixel) and ``image_path`` plus
    several required patch-related fields. ``overlap`` defaults to ``-1``.
    """

    mpp: float
    level: int
    patch_size: int
    patch_stride: int | None = None
    patch_mpp: float
    target_mpp: float
    scale_factor: float
    overlap: float = -1


class XeniumCoordinate(SlideCoordinate):
    """Coordinate that references an external file containing point data."""

    points_path: str
