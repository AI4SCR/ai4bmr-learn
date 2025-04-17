

def get_thumbnail_size_and_scale(size, max_size: int = 1000):
    h, w = size
    scale = max_size / max(h, w)
    size = (round(h * scale), round(w * scale))
    return size, scale


class Patcher:

    def __init__(self, kernel_size: int, stride: int | None = None):
        self.kernel_size = kernel_size
        self.stride = stride or self.kernel_size

    def get_coordinates(self, height: int, width: int, **kwargs):
        import numpy as np
        from ai4bmr_learn.data_models.Coordinate import BaseCoordinate

        x_coords = np.arange(0, width - self.kernel_size + 1, self.stride)
        y_coords = np.arange(0, height - self.kernel_size + 1, self.stride)

        coords = [BaseCoordinate(x=x, y=y, kernel_size=self.kernel_size, stride=self.stride, **kwargs)
                  for y in y_coords for x in x_coords]
        return coords
