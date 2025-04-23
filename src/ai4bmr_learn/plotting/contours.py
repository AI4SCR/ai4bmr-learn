import matplotlib.pyplot as plt
import openslide
from ai4bmr_learn.data_models.Coordinate import BaseCoordinate
from ai4bmr_learn.utils.images import get_thumbnail, get_thumbnail_size_and_scale
from shapely.affinity import scale
import geopandas as gpd
import numpy as np
from PIL.Image import Image


def draw_contours(canvas: np.ndarray,
                  contours: gpd.GeoDataFrame,
                  scale_factor: float = 1,
                  thickness: int = 1,
                  engine: str = "cv2",
                  ) -> plt.Axes:

    contours = contours["geometry"].apply(
        lambda geom: scale(geom, xfact=scale_factor, yfact=scale_factor, origin=(0, 0)))

    if engine == "matplotlib":
        ax = plt.imshow(canvas).axes
        contours.plot(ax=ax, facecolor="none", edgecolor='green', linewidth=thickness)
        return ax

    elif engine == 'cv2':
        import cv2
        contours_ = [np.array(geom.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
                     for geom in contours.geometry]
        cv2.drawContours(canvas, contours_, -1, (0, 255, 0), thickness)
        return canvas

    else:
        raise ValueError(f"Unknown engine: {engine}. Use 'matplotlib' or 'cv2'.")


def visualize_contours(contours, *, slide: openslide.OpenSlide, image = None, max_size: int = 1000, thickness: int = 1):

    if image is None:
        canvas, scale_factor = get_thumbnail(slide=slide, image=image, max_size=max_size)
    else:
        height, width, _ = image.shape
        max_size = max(height, width)
        size = slide.level_dimensions[0]
        _, scale_factor = get_thumbnail_size_and_scale(size, max_size=max_size)
        canvas = image.copy()

    return draw_contours(
        canvas=canvas,
        contours=contours,
        scale_factor=scale_factor,
        thickness=thickness,
        engine='cv2'
    )
