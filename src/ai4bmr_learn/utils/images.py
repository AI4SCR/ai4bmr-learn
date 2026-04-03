import numpy as np
import openslide


def get_thumbnail_size_and_scale(size, max_size: int = 1000):
    import numpy as np

    h, w = size

    if max(h, w) <= max_size:
        return size, 1

    scale = max_size / max(h, w)
    # note: Openslide.get_thumbnail() uses the largest downsample factor, thus we need to increase the size of the
    # smaller dimension
    # TODO: maybe add flag to use round instead of ceil
    # size = (round(h * scale), round(w * scale))
    size = int(np.ceil(h * scale)), int(np.ceil(w * scale))
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
