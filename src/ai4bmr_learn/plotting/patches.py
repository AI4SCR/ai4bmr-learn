import geopandas as gpd
import numpy as np
from PIL.Image import Image


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
