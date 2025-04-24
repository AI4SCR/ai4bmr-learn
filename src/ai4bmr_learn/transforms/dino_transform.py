# Adapted from: lightly.transforms.dino_transform.DINOTransform

from typing import Dict, List, Optional, Tuple, Union

import PIL
import torch.nn as nn
from PIL.Image import Image
from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.rotation import random_rotation_transform
from lightly.transforms.solarize import RandomSolarization
from lightly.transforms.torchvision_v2_compatibility import torchvision_transforms as T
from torchvision.transforms import v2
from lightly.transforms.utils import IMAGENET_NORMALIZE
from torch import Tensor


class DINOTransform(nn.Module):
    """Implements the global and local view augmentations for DINO [0].

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of Tensor of length 2 * global + n_local_views. (8 by default)

    Applies the following augmentations by default:
        - Random resized crop
        - Random horizontal flip
        - Color jitter
        - Random gray scale
        - Gaussian blur
        - Random solarization
        - ImageNet normalization

    This class generates two global and a user defined number of local views
    for each image in a batch. The code is adapted from [1].

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294
    - [1]: https://github.com/facebookresearch/dino

    Attributes:
        global_crop_size:
            Crop size of the global views.
        global_crop_scale:
            Tuple of min and max scales relative to global_crop_size.
        local_crop_size:
            Crop size of the local views.
        local_crop_scale:
            Tuple of min and max scales relative to local_crop_size.
        n_local_views:
            Number of generated local views.
        hf_prob:
            Probability that horizontal flip is applied.
        vf_prob:
            Probability that vertical flip is applied.
        rr_prob:
            Probability that random rotation is applied.
        rr_degrees:
            Range of degrees to select from for random rotation. If rr_degrees is None,
            images are rotated by 90 degrees. If rr_degrees is a (min, max) tuple,
            images are rotated by a random angle in [min, max]. If rr_degrees is a
            single number, images are rotated by a random angle in
            [-rr_degrees, +rr_degrees]. All rotations are counter-clockwise.
        cj_prob:
            Probability that color jitter is applied.
        cj_strength:
            Strength of the color jitter. `cj_bright`, `cj_contrast`, `cj_sat`, and
            `cj_hue` are multiplied by this value.
        cj_bright:
            How much to jitter brightness.
        cj_contrast:
            How much to jitter constrast.
        cj_sat:
            How much to jitter saturation.
        cj_hue:
            How much to jitter hue.
        random_gray_scale:
            Probability of conversion to grayscale.
        gaussian_blur:
            Tuple of probabilities to apply gaussian blur on the different
            views. The input is ordered as follows:
            (global_view_0, global_view_1, local_views)
        kernel_size:
            Size of the gaussian kernel.
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
        solarization:
            Probability to apply solarization on the second global view.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    """

    def __init__(
            self,
            global_crop_size: int = 224,
            global_crop_scale: Tuple[float, float] = (0.4, 1.0),
            local_crop_size: int = 96,
            local_crop_scale: Tuple[float, float] = (0.05, 0.4),
            n_local_views: int = 6,
            hf_prob: float = 0.5,
            vf_prob: float = 0,
            rr_prob: float = 0,
            rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
            cj_prob: float = 0.8,
            cj_strength: float = 0.5,
            cj_bright: float = 0.8,
            cj_contrast: float = 0.8,
            cj_sat: float = 0.4,
            cj_hue: float = 0.2,
            random_gray_scale: float = 0.2,
            gaussian_blur: Tuple[float, float, float] = (1.0, 0.1, 0.5),
            kernel_size: int = 5,
            sigmas: Tuple[float, float] = (0.1, 2),
            solarization_prob: float = 0.2,
            normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        super().__init__()

        # TODO:
        # compute the kernel size form the sigma, however, we do not know the sigma in advance
        # kernel_size = 2 * int(4 * s + 0.5) + 1
        # another issue is that the kernel size should maybe be different for local and global crops

        # first global crop
        global_transform_0 = DINOViewTransform(
            crop_size=global_crop_size,
            crop_scale=global_crop_scale,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            rr_prob=rr_prob,
            rr_degrees=rr_degrees,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_hue=cj_hue,
            cj_sat=cj_sat,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur[0],
            kernel_size=kernel_size,
            sigmas=sigmas,
            solarization_prob=0,
            normalize=normalize,
        )

        # second global crop
        global_transform_1 = DINOViewTransform(
            crop_size=global_crop_size,
            crop_scale=global_crop_scale,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            rr_prob=rr_prob,
            rr_degrees=rr_degrees,
            cj_prob=cj_prob,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_hue=cj_hue,
            cj_sat=cj_sat,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur[1],
            kernel_size=kernel_size,
            sigmas=sigmas,
            solarization_prob=solarization_prob,
            normalize=normalize,
        )

        # transformation for the local small crops
        local_transform = DINOViewTransform(
            crop_size=local_crop_size,
            crop_scale=local_crop_scale,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            rr_prob=rr_prob,
            rr_degrees=rr_degrees,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_hue=cj_hue,
            cj_sat=cj_sat,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur[2],
            kernel_size=kernel_size,
            sigmas=sigmas,
            solarization_prob=0,
            normalize=normalize,
        )
        self.local_transforms = [local_transform] * n_local_views
        self.global_transforms = [global_transform_0, global_transform_1]

    def forward(self, item: dict):
        image = item["image"]

        local_views = [transform(image) for transform in self.local_transforms]
        global_views = [transform(image) for transform in self.global_transforms]

        item['local_views'] = local_views
        item['global_views'] = global_views

        return item


class DINOViewTransform(nn.Module):
    def __init__(
            self,
            crop_size: int = 224,
            crop_scale: Tuple[float, float] = (0.4, 1.0),
            hf_prob: float = 0.5,
            vf_prob: float = 0,
            rr_prob: float = 0,
            rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
            cj_prob: float = 0.8,
            cj_strength: float = 0.5,
            cj_bright: float = 0.8,
            cj_contrast: float = 0.8,
            cj_sat: float = 0.4,
            cj_hue: float = 0.2,
            random_gray_scale: float = 0.2,
            gaussian_blur: float = 1.0,
            kernel_size: int = 5,
            sigmas: Tuple[float, float] = (0.1, 2),
            solarization_prob: float = 0.2,
            normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        super().__init__()

        transform = [
            v2.RandomResizedCrop(
                size=crop_size,
                scale=crop_scale,
                # Type ignore needed because BICUBIC is not recognized as an attribute.
                interpolation=PIL.Image.BICUBIC,  # type: ignore[attr-defined]
            ),
            v2.RandomHorizontalFlip(p=hf_prob),
            v2.RandomVerticalFlip(p=vf_prob),
            # random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            v2.RandomApply(
                [
                    v2.RandomRotation(degrees=rr_degrees),
                ],
                p=rr_prob,
            ),
            v2.RandomApply(
                [
                    T.ColorJitter(
                        brightness=cj_strength * cj_bright,
                        contrast=cj_strength * cj_contrast,
                        saturation=cj_strength * cj_sat,
                        hue=cj_strength * cj_hue,
                    )
                ],
                p=cj_prob,
            ),
            v2.RandomGrayscale(p=random_gray_scale),
            v2.RandomApply(
                [
                    v2.GaussianBlur(
                        kernel_size=kernel_size,
                        sigma=sigmas,
                    )
                ],
                p=gaussian_blur,
            ),
            # GaussianBlur(
            #     kernel_size=kernel_size,
            #     sigmas=sigmas,
            #     prob=gaussian_blur,
            # ),
            v2.RandomSolarize(p=solarization_prob),
            # T.RandomSolarization(prob=solarization_prob),
            # T.ToTensor(),
        ]

        if normalize:
            transform += [v2.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = v2.Compose(transform)

    def forward(self, item: dict) -> Tensor:
        transformed: Tensor = self.transform(item)
        return transformed
