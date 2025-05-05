from shapely.affinity import scale, translate
from torch import are_deterministic_algorithms_enabled, nn
from torchvision.transforms import v2
import torch
from ai4bmr_learn.utils.utils import pair
import math

class RandomCrop(nn.Module):
    def __init__(self, scale: tuple[float, float] = (0.4, 1.0)):
        super().__init__()
        self.scale = scale

    def make_params(self, height, width):
        aspect_ratio = 1
        area = height * width

        target_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        i = torch.randint(0, height - h + 1, size=(1,)).item()
        j = torch.randint(0, width - w + 1, size=(1,)).item()

        return dict(left=j, top=i, height=h, width=w)

    def forward(self, item):
        result = {k:v for k,v in item.items()}

        height, width = pair(item['kernel_size'])
        params = self.make_params(height=height, width=width)

        # transform points
        xmin = params['left']
        xmax = params['left'] + params['width']
        ymin = params['top']
        ymax = params['top'] + params['height']

        points = item['points']
        filter_ = points.geometry.x.between(xmin, xmax) & points.geometry.y.between(ymin, ymax)
        points = points[filter_]

        xoff = -params['left']
        yoff = -params['top']

        points['geometry'] = points['geometry'].map(
            lambda geom: translate(geom, xoff=xoff, yoff=yoff)
        )

        height = params['height']
        width = params['width']

        assert points.geometry.x.min() >= 0
        assert points.geometry.y.min() >= 0
        assert points.geometry.x.max() <= width
        assert points.geometry.y.max() <= height

        result['points'] = points
        result['height'] = height
        result['width'] = width

        return result
    