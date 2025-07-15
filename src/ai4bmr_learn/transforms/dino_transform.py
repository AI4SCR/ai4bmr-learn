import PIL
import torch.nn as nn
from torch import Tensor
from torchvision.transforms import v2
from torchvision import tv_tensors
from torchvision.transforms.functional import InterpolationMode

class DINOTransformLightly:
    """Wrapper to support dict items with lightly DINOTransform"""
    def __init__(self, **kwargs):
        from lightly.transforms import DINOTransform
        self.transform = DINOTransform(**kwargs)

    def __call__(self, item: dict):
        from torchvision.transforms.functional import to_pil_image
        img = item['image']

        if not isinstance(img, PIL.Image.Image):
            img = to_pil_image(img)

        views = [{**item, 'image': view} for view in self.transform(img)]

        return {'global_views': views[:2], 'local_views': views[2:]}


class DINOTransform(nn.Module):
    """DINO transforms for dict items with tv_tensor.Image"""

    def __init__(
            self,
            num_local_views: int = 6,
            global_crop_size: int = 224,
            global_crop_scale: tuple[float, float] = (0.4, 1.0),
            local_crop_size: int = 96,
            local_crop_scale: tuple[float, float] = (0.25, 0.4),
            hf_prob: float = 0.25,
            vf_prob: float = 0.25,
            # rr_prob: float = 0.25,
            # rr_degrees: tuple[float] = (90, 180, 270),
            gaussian_blur_probs: tuple[float, float, float] = (1., .1, .5),
            sigmas: tuple[float, float] = (0.1, 2.0),
    ):
        super().__init__()

        global_transform_0 = DINOViewTransform(
            crop_size=global_crop_size,
            crop_scale=global_crop_scale,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            # rr_prob=rr_prob,
            # rr_degrees=rr_degrees,
            gaussian_blur_prob=gaussian_blur_probs[0],
            sigmas=sigmas,
        )

        global_transform_1 = DINOViewTransform(
            crop_size=global_crop_size,
            crop_scale=global_crop_scale,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            # rr_prob=rr_prob,
            # rr_degrees=rr_degrees,
            gaussian_blur_prob=gaussian_blur_probs[1],
            sigmas=sigmas,
        )

        local_transform = DINOViewTransform(
            crop_size=local_crop_size,
            crop_scale=local_crop_scale,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            # rr_prob=rr_prob,
            # rr_degrees=rr_degrees,
            gaussian_blur_prob=gaussian_blur_probs[2],
            sigmas=sigmas,
        )

        self.local_transforms = [local_transform] * num_local_views
        self.global_transforms = [global_transform_0, global_transform_1]

    def forward(self, item: dict):
        # collect the items that need to undergo geometric transformations and remove original items
        i = {k: v for k, v in item.items() if isinstance(v, (tv_tensors.Image, tv_tensors.Mask))}
        for k in i:
            del item[k]

        local_views = [transform(i) for transform in self.local_transforms]
        global_views = [transform(i) for transform in self.global_transforms]

        item['local_views'] = local_views
        item['global_views'] = global_views

        return item


class DINOViewTransform(nn.Module):
    def __init__(
            self,
            crop_size: int = 224,
            crop_scale: tuple[float, float] = (0.25, 0.4),
            hf_prob: float = 0.25,
            vf_prob: float = 0.25,
            # rr_prob: float = 0.25,
            # rr_degrees: tuple[float] = (90, 180, 270),
            gaussian_blur_prob: float = 0.25,
            sigmas: tuple[float, float] = (0.1, 2.0),
    ):
        super().__init__()

        kernel_size = int(crop_size * 0.1)
        kernel_size = kernel_size + 1 * (kernel_size + 1) % 2  # kernel_size needs to be odd

        transform = [
            v2.RandomResizedCrop(
                size=crop_size,
                scale=crop_scale,
                interpolation=InterpolationMode.BICUBIC
            ),
            v2.RandomHorizontalFlip(p=hf_prob),
            v2.RandomVerticalFlip(p=vf_prob),
            # TODO: allow only discrete rotation values
            # v2.RandomApply(
            #     [
            #         v2.RandomRotation(degrees=rr_degrees, interpolation=InterpolationMode.BICUBIC)
            #     ],
            #     p=rr_prob,
            # ),
            v2.RandomApply(
                [
                    v2.GaussianBlur(kernel_size=kernel_size, sigma=sigmas)
                ],
                p=gaussian_blur_prob,
            ),
        ]

        self.transform = v2.Compose(transform)

    def forward(self, item: dict) -> Tensor:
        transformed: Tensor = self.transform(item)
        return transformed