import matplotlib.pyplot as plt
import openslide
from PIL.ImageChops import add_modulo

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
                  color_by_contour: bool = False,
                  add_label: bool = False,
                  ) -> plt.Axes | np.ndarray:
    contours = contours["geometry"].apply(
        lambda geom: scale(geom, xfact=scale_factor, yfact=scale_factor, origin=(0, 0)))

    if color_by_contour:
        from ai4bmr_learn.plotting.utils import get_colorcet_map
        color_map = get_colorcet_map(contours.index, as_int=True)
    else:
        color_map = {}

    if engine == "matplotlib":
        ax = plt.imshow(canvas).axes
        contours.plot(ax=ax, facecolor="none", edgecolor='green', linewidth=thickness)
        return ax

    elif engine == 'cv2':
        import cv2
        contours_ = [np.array(geom.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
                     for geom in contours.geometry]
        colors = [color_map.get(i, (0, 255, 0)) for i in contours.index]
        for contour, color in zip(contours_, colors):
            cv2.drawContours(canvas, [contour], -1, color, thickness)

        if add_label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1

            for i, geom in zip(contours.index, contours.geometry):
                centroid = geom.centroid
                x, y = int(centroid.x), int(centroid.y)
                text_color = color_map.get(i, (0, 255, 0))
                cv2.putText(canvas, str(i), (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        return canvas

    else:
        raise ValueError(f"Unknown engine: {engine}. Use 'matplotlib' or 'cv2'.")


def visualize_contours(contours, *, slide: openslide.OpenSlide, image=None, max_size: int = 1000,
                       thickness: int = 1, color_by_contour: bool = False, add_label: bool = False) -> np.ndarray:
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
        engine='cv2',
        add_label=add_label,
        color_by_contour=color_by_contour
    )
