import numpy as np
import openslide


def get_thumbnail_size_and_scale(size, max_size: int = 1000):
    h, w = size
    scale = max_size / max(h, w)
    size = (round(h * scale), round(w * scale))
    return size, scale

def get_thumbnail(*, slide: openslide.OpenSlide = None, image: np.ndarray = None, max_size: int = 1000):
    if slide is not None:
        size = slide.level_dimensions[0][:2]
        size, _ = get_thumbnail_size_and_scale(size, max_size=max_size)
        return slide.get_thumbnail(size=size)

    if image is not None:
        import cv2
        (h, w), scale = get_thumbnail_size_and_scale(image.shape[:2], max_size=max_size)
        return cv2.resize(region, (w, h), interpolation=cv2.INTER_AREA)

def get_patcher_params(slide, kernel_size: int, stride: int, target_mpp: float):
    from ai4bmr_learn.utils.slides import get_mpp_and_resolution
    width, height = slide.dimensions
    image_path = str(slide._filename)

    mpp, res = get_mpp_and_resolution(slide)
    scale_factor = target_mpp / mpp

    new_kernel_size = int(kernel_size * scale_factor)
    new_stride = int(stride * scale_factor)

    effective_scale_factor = new_kernel_size / kernel_size
    effective_mpp = mpp * effective_scale_factor
    return dict(
        height=height,
        width=width,
        kernel_size=new_kernel_size,
        stride=new_stride,
        mpp=mpp,
        target_mpp=target_mpp,
        effective_mpp=effective_mpp,
        scale_factor=effective_scale_factor,
        image_path=image_path,
    )


def get_coordinates(height: int, width: int, kernel_size: int, stride: int, **kwargs):
    import numpy as np
    from ai4bmr_learn.data_models.Coordinate import SlideCoordinate

    x_coords = np.arange(0, width - kernel_size + 1, stride)
    y_coords = np.arange(0, height - kernel_size + 1, stride)

    coords = [SlideCoordinate(x=int(x), y=int(y), kernel_size=kernel_size, stride=stride, **kwargs)
              for y in y_coords for x in x_coords]
    return coords

from ai4bmr_learn.data_models.Coordinate import BaseCoordinate, SlideCoordinate
def coord_to_bbox(coord: BaseCoordinate | SlideCoordinate):
    from shapely.geometry import box
    x = coord.x
    y = coord.y
    kernel_size = coord.kernel_size

    xmin, xmax = x, x + kernel_size
    ymin, ymax = y, y + kernel_size

    bbox_coords = [xmin, ymin, xmax, ymax]
    bbox = box(*bbox_coords)
    return bbox

import geopandas as gpd
def filter_coords(coords: list[SlideCoordinate], *, contours: gpd.GeoDataFrame, overlap: float = 0.25):
    filtered = []
    for coord in coords:
        bbox = coord_to_bbox(coord)
        overlap_by_contour = contours.geometry.intersection(bbox).area / bbox.area
        if overlap_by_contour.sum() >= overlap:
            coord.overlap = overlap_by_contour.sum()
            filtered.append(coord)

    return filtered
