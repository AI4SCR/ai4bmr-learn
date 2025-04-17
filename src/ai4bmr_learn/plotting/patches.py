import matplotlib.pyplot as plt
from ai4bmr_learn.data_models.Coordinate import BaseCoordinate
import geopandas as gpd
import numpy as np

def draw_coords(
    canvas: np.ndarray,
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

def draw_contours(canvas: np.ndarray,
                  contours: gpd.GeoDataFrame,
                  scale_factor: float = 1,
                  linewidth: int = 1,
                  ) -> plt.Axes:
    from shapely.affinity import scale

    contours = contours["geometry"].apply(lambda geom: scale(geom, xfact=scale_factor, yfact=scale_factor, origin=(0, 0)))
    ax = plt.imshow(canvas).axes

    contours.plot(ax=ax, facecolor="none", edgecolor='green', linewidth=1)

    # import cv2
    # cv2.drawContours(canvas, contours, -1, color, thickness)
    return ax
