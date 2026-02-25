
import geopandas as gpd
import numpy as np
import torch
from torchvision.transforms import v2
from ai4bmr_learn.utils.helpers import pair

class RandomResizeCrop:  # this doesn't have to be a nn.Module
    def __init__(self, errors: str = 'clip-raise', **kwargs):
        self.random_resize_crop = v2.RandomResizedCrop(**kwargs)
        self.errors = errors

    def __call__(self, item: dict) -> dict:
        # NOTE: we cannot overwrite the `transform` call as suggested by the v2 documentation.
        #   the reason is that only torch.tensors, Images, Masks,... are passed to transform.
        #   and I could not find how one could configure the transform to accept other types without hacking the types.

        # FIXME: do we overwrite the input batch or do we need to create a copy?
        result = item.copy()

        image = None
        if 'image' in item:
            image = item['image']

        if 'patch_size' in item:
            patch_size = item['patch_size']

            img_shape = pair(patch_size)
            if image is not None:
                assert image.shape[1:] == img_shape
            else:
                image = torch.zeros(img_shape)

            item['patch_size'] = self.random_resize_crop.size

        # NOTE: we need to create the params ourselves to be able to apply them to the `points` data
        params = self.random_resize_crop.make_params(image)

        if 'image' in item:
            result['image'] = self.random_resize_crop.transform(image, params=params)

        if 'points' in item:
            pts = item['points']
            if not len(pts):
                result['points'] = pts
                result.setdefault('metadata', {}).setdefault('transform', {})['RandomResizeCrop'] = params
                return result

            # pull coords once
            xs = pts.geometry.x.to_numpy(copy=False)
            ys = pts.geometry.y.to_numpy(copy=False)

            # bbox filter
            xmin, ymin = params['left'], params['top']
            xmax = xmin + params['width']
            ymax = ymin + params['height']
            mask = (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)

            if mask.any():
                # slice once; copy=True so we don’t modify parent df
                pts_crop = pts.loc[mask].copy()

                # translate + scale
                dst_h, dst_w = self.random_resize_crop.size

                xoff, yoff = -xmin, -ymin
                xfact = dst_w / params['width']
                yfact = dst_h / params['height']

                xs_m = (xs[mask] + xoff) * xfact
                ys_m = (ys[mask] + yoff) * yfact

                if 'clip' in self.errors:
                    xs_m = np.clip(xs_m, 0, dst_w)
                    ys_m = np.clip(ys_m, 0, dst_h)

                if 'raise' in self.errors:
                    atol = 1e-6
                    xmin, ymin = xs_m.min(), ys_m.min()
                    xmax, ymax = xs_m.max(), ys_m.max()

                    if xmin < 0:
                        assert np.isclose(xmin, 0, rtol=0, atol=atol), f'xmin={xmin}'
                    if ymin < 0:
                        assert np.isclose(ymin, 0, rtol=0, atol=atol), f'ymin={ymin}'
                    if xmax > dst_w:
                        assert np.isclose(xmax, dst_w, rtol=0, atol=atol), f'xmax={xmax} > {dst_w}'
                    if ymax > dst_h:
                        assert np.isclose(ymax, dst_h, rtol=0, atol=atol), f'ymax={ymax} > {dst_h}'

                # rebuild geometry in one go
                pts_crop['geometry'] = gpd.points_from_xy(xs_m, ys_m)
                result['points'] = pts_crop
            else:
                # no points survive
                result['points'] = pts.iloc[[]]

        result.setdefault('metadata', {}).setdefault('transform', {})['RandomResizeCrop'] = params
        return result