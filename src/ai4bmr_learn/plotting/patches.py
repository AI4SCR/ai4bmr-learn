from ai4bmr_learn.data_models.Coordinate import BaseCoordinate
from ai4bmr_learn.utils.images import get_thumbnail_size_and_scale

import numpy as np
from PIL.Image import Image

def draw_coords(
    canvas: np.ndarray | Image,
    coords: list[BaseCoordinate],
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


def visualize_coords(slide, coords, canvas = None, level=0, max_size=1000):
    size = slide.level_dimensions[level][:2]
    size, scale_factor = get_thumbnail_size_and_scale(size=size, max_size=1000)

    if canvas is None:
        canvas = slide.get_thumbnail(size=size)
        canvas = np.asarray(canvas).copy()
    else:
        assert max(canvas.shape[:2]) == max_size

    return draw_coords(
        canvas=canvas,
        coords=coords,
        scale_factor=scale_factor,
        thickness=1,
    )