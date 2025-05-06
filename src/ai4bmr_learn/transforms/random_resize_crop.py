from shapely.affinity import scale, translate
from torch import nn
from torchvision.transforms import v2


class RandomResizeCrop(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.random_resize_crop = v2.RandomResizedCrop(**kwargs)

    def forward(self, item):
        # NOTE: we cannot overwrite the `transform` call as suggested by the v2 documentation.
        #   the reason is that only torch.tensors, Images, Masks,... are passed to transform.
        #   and I could not find how one could configure the transform to accept other types without hacking the types.

        # FIXME: do we overwrite the input batch or do we need to create a copy?
        result = {k:v for k,v in item.items()}

        # NOTE: we need to create the params ourselves to be able to apply them to the `points` data
        image = item['image']
        _, height, width = image.shape

        params = self.random_resize_crop.make_params(image)
        result['image'] = self.random_resize_crop.transform(image, params=params)

        # transform points
        xmin = params['left']
        xmax = params['left'] + params['width']
        ymin = params['top']
        ymax = params['top'] + params['height']

        points = item['points']
        filter_ = points.geometry.x.between(xmin, xmax) & points.geometry.y.between(ymin, ymax)
        points = points[filter_]

        # scale/translate points to match the crop after resizing
        xfact = self.random_resize_crop.size[0] / params['width']
        yfact = self.random_resize_crop.size[1] / params['height']

        xoff = -params['left']
        yoff = -params['top']

        points = points.assign(geometry=points['geometry'].map(
            lambda geom: scale(
                translate(geom, xoff=xoff, yoff=yoff),
                xfact=xfact,
                yfact=yfact,
                origin=(0, 0)
            )
        ))

        assert points.geometry.y.min() >= 0
        assert points.geometry.x.min() >= 0
        assert points.geometry.y.max() <= self.random_resize_crop.size[0]
        assert points.geometry.x.max() <= self.random_resize_crop.size[1]

        result['points'] = points
        result.setdefault('metadata', {}).setdefault('transform', {})['RandomResizeCrop'] = params

        return result
    