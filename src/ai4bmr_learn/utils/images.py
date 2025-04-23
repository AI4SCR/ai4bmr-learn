import numpy as np
import openslide
from ai4bmr_learn.utils.utils import pair

def get_thumbnail_size_and_scale(size, max_size: int = 1000):
    h, w = size
    
    if max(h, w) <= max_size:
        return size, 1
    
    scale = max_size / max(h, w)
    size = (round(h * scale), round(w * scale))
    return size, scale

def get_thumbnail(*, slide: openslide.OpenSlide = None, image: np.ndarray = None, max_size: int = 1000) -> tuple[np.ndarray, float]:
    if slide is not None:
        size = slide.level_dimensions[0][:2]
        size, scale_factor = get_thumbnail_size_and_scale(size, max_size=max_size)
        thumbnail = slide.get_thumbnail(size=size)
        return np.asarray(thumbnail).copy(), scale_factor

    if image is not None:
        import cv2
        (h, w), scale_factor = get_thumbnail_size_and_scale(image.shape[:2], max_size=max_size)
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA), scale_factor


def get_slide_patcher_params(slide, patch_size: int, patch_stride: int, target_mpp: float):
    from ai4bmr_learn.utils.slides import get_mpp_and_resolution
    width, height = slide.dimensions
    image_path = str(slide._filename)

    mpp, res = get_mpp_and_resolution(slide)
    scale_factor = target_mpp / mpp

    kernel_size = round(patch_size * scale_factor)
    stride = round(patch_stride * scale_factor)

    effective_scale_factor = kernel_size / patch_size
    effective_mpp = mpp * effective_scale_factor
    return dict(
        height=height,
        width=width,
        
        kernel_size=kernel_size,
        stride=stride,
        mpp=mpp,
        
        patch_size=patch_size,
        patch_stride=patch_stride,
        patch_mpp=effective_mpp,
        target_mpp=target_mpp,
        scale_factor=effective_scale_factor,
        
        image_path=image_path,
    )


def get_coordinates_dict(height: int, width: int, kernel_size: int | tuple[int, int], stride: int | tuple[int, int], **kwargs) -> dict:
    import numpy as np
    from itertools import product

    kh, kw = pair(kernel_size)
    sh, sw = pair(stride)

    x_coords = np.arange(0, width - kw + 1, sw)
    y_coords = np.arange(0, height - kh + 1, sh)

    coords = product(y_coords, x_coords)
    coords = [dict(id=i, x=int(x), y=int(y), kernel_size=kernel_size, stride=stride, **kwargs)
              for i, (y, x) in enumerate(coords)]
    return coords

from ai4bmr_learn.data_models.Coordinate import BaseCoordinate, SlideCoordinate, XeniumCoordinate
def coord_to_bbox(coord: BaseCoordinate | SlideCoordinate | XeniumCoordinate):
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
def filter_coords(coords: list[SlideCoordinate | XeniumCoordinate], *, contours: gpd.GeoDataFrame, overlap: float = 0.25):
    filtered = []
    for coord in coords:
        bbox = coord_to_bbox(coord)
        overlap_by_contour = contours.geometry.intersection(bbox).area / bbox.area
        if overlap_by_contour.sum() >= overlap:
            coord.overlap = overlap_by_contour.sum()
            filtered.append(coord)

    return filtered


def get_patch(coord):
    import openslide
    from torchvision.transforms import v2
    import torch

    img_path = coord.image_path
    slide = openslide.OpenSlide(img_path)

    x, y = coord.x, coord.y
    kernel_height, kernel_width = pair(coord.kernel_size)
    patch_height, patch_width = pair(coord.patch_size)
    scale_factor = coord.scale_factor

    assert patch_height == round(kernel_height / scale_factor)
    assert patch_width == round(kernel_width / scale_factor)

    # MORPHOLOGY
    patch = slide.read_region((x, y), 0, (kernel_width, kernel_height))
    patch = patch.convert("RGB")  # remove alpha channel, #TODO: is this more efficient than current solution?
    # note: this order seems to perform the best, i.e. resize after ToImage
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((patch_height, patch_width)),
        # v2.ToDtype(torch.float32, scale=True),
    ])
    patch = transform(patch)

    return patch

def get_points(coord):
    from shapely.affinity import translate, scale
    from ai4bmr_learn.utils.utils import pair

    kernel_height, kernel_width = pair(coord.kernel_size)
    
    xmin = coord.x
    ymin = coord.y
    xmax = xmin + kernel_width
    ymax = ymin + kernel_height
    bbox = (xmin, ymin, xmax, ymax)

    scale_factor = coord.scale_factor

    points = gpd.read_file(coord.points_path, bbox=bbox)
    # points['geometry'] = points['geometry'].map(lambda geom: translate(geom=geom, xoff=-xmin, yoff=-ymin, zoff=0))
    # points['geometry'] = points['geometry'].map(lambda geom: scale(geom=geom, xfact=scale_factor, yfact=scale_factor, zoff=0))

    points['geometry'] = points['geometry'].map(
        lambda geom: scale(
            translate(geom=geom, xoff=-xmin, yoff=-ymin, zoff=0),
            xfact=1 / scale_factor,
            yfact=1 / scale_factor,
            origin=(0, 0)
        )
    )

    return points