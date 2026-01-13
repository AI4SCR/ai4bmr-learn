from ai4bmr_learn.data_models.Coordinate import PatchCoordinate
from ai4bmr_learn.utils.images import get_thumbnail, get_thumbnail_size_and_scale
import openslide

import numpy as np
from PIL.Image import Image


def draw_coords(
        canvas: np.ndarray | Image,
        coords: list[PatchCoordinate],
        scale_factor: float = 1,
        thickness: int | None = None,
):
    import cv2

    thickness = thickness or min(canvas.shape[:2]) / 100
    thickness = max(thickness, 1)
    for item in coords:
        y = int(item.y * scale_factor)
        x = int(item.x * scale_factor)
        patch_size = int(item.kernel_size * scale_factor)

        color = (255, 0, 0)
        canvas = cv2.rectangle(
            canvas, (x, y), (x + patch_size, y + patch_size), color, thickness
        )

    return canvas


def visualize_coords(coords, *, slide: openslide.OpenSlide, image=None, max_size: int = 1000, thickness: int = 1):
    if image is None:
        canvas, scale_factor = get_thumbnail(slide=slide, image=image, max_size=max_size)
    else:
        height, width, _ = image.shape
        max_size = max(height, width)
        size = slide.level_dimensions[0]
        _, scale_factor = get_thumbnail_size_and_scale(size, max_size=max_size)
        canvas = image.copy()

    return draw_coords(
        canvas=canvas,
        coords=coords,
        scale_factor=scale_factor,
        thickness=thickness,
    )


import geopandas as gpd


def draw_tiles(
        canvas: np.ndarray | Image,
        tiles: gpd.GeoDataFrame,
        scale_factor: float = 1,
        thickness: int | None = None,
        color = (128, 128, 128),
        alpha: float = 0.5,
):
    import cv2

    thickness = thickness or min(canvas.shape[:2]) / 100
    thickness = max(thickness, 1)

    overlay = canvas.copy()

    for i in range(len(tiles)):
        tile = tiles.iloc[i]
        xmin, ymin, xmax, ymax = tile.geometry.bounds

        tile_width = int(xmax - xmin)
        tile_height = int(ymax - ymin)
        assert tile_width == tile_height, "Only square tiles are supported"

        y = int(ymin * scale_factor)
        x = int(xmin * scale_factor)
        patch_size = int(tile_width * scale_factor)

        cv2.rectangle(
            overlay,
            (x, y),
            (x + patch_size, y + patch_size),
            color,
            thickness,
        )

    canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

    return canvas
