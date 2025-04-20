import matplotlib.pyplot as plt
from ai4bmr_learn.data_models.Coordinate import BaseCoordinate
from ai4bmr_learn.utils.images import get_thumbnail_size_and_scale
from shapely.affinity import scale
import geopandas as gpd
import numpy as np
from PIL.Image import Image


def draw_contours(canvas: np.ndarray,
                  contours: gpd.GeoDataFrame,
                  scale_factor: float = 1,
                  linewidth: int = 1,
                  engine: str = "cv2",
                  ) -> plt.Axes:
    contours = contours["geometry"].apply(
        lambda geom: scale(geom, xfact=scale_factor, yfact=scale_factor, origin=(0, 0)))

    if engine == "matplotlib":
        ax = plt.imshow(canvas).axes
        contours.plot(ax=ax, facecolor="none", edgecolor='green', linewidth=1)
        return ax
    elif engine == 'cv2':
        import cv2
        contours_ = [np.array(geom.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
                     for geom in contours.geometry]
        cv2.drawContours(canvas, contours_, -1, (0, 255, 0), linewidth)
        return canvas
    else:
        raise ValueError(f"Unknown engine: {engine}. Use 'matplotlib' or 'cv2'.")


def visualize_contours(slide, contours, canvas=None, level=0, max_size=1000):
    size = slide.level_dimensions[level][:2]
    size, scale_factor = get_thumbnail_size_and_scale(size=size, max_size=1000)

    if canvas is None:
        canvas = slide.get_thumbnail(size=size)
        canvas = np.asarray(canvas).copy()
    else:
        assert max(canvas.shape[:2]) == max_size

    return draw_contours(
        canvas=canvas,
        contours=contours,
        scale_factor=scale_factor,
        linewidth=1,
        engine='cv2'
    )
