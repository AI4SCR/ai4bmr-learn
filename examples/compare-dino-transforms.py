from ai4bmr_learn.transforms.dino_transform import DINOTransformLightly
from ai4bmr_learn.transforms.dino_transformv2 import DINOTransform
from ai4bmr_learn.datasets.vocdetection import VOCDetection
from matplotlib import pyplot as plt
import numpy as np

transform_lightly = DINOTransformLightly(local_crop_scale=(0.25, 0.4),
                                         hf_prob=0, vf_prob=0, rr_prob=0, cj_prob=0.0, random_gray_scale=0,
                                         solarization_prob=0, normalize=None)

transform = DINOTransform(local_crop_scale=(0.25, 0.4), hf_prob=0, vf_prob=0, rr_prob=0)


def target_transform(t):
    return 0

ds_lightly = VOCDetection(transform=transform_lightly, target_transform=target_transform)
ds = VOCDetection(transform=transform, target_transform=target_transform)

# check that transform does not yield nan
for i in range(1000):
    item = ds[i]
    for view in item['global_views'] + item['local_views']:
        img = view['image'].permute(1,2,0).numpy()
        assert not np.isnan(img).any()


for i in range(100):
    fig, axs_grid = plt.subplots(2, 8)
    for item, axs in zip([ds_lightly[i], ds[i]], axs_grid):
        for view, ax in zip(item['global_views'] + item['local_views'], axs):
            img = view['image'].permute(1,2,0).numpy()
            assert not np.isnan(img).any()

            ax.imshow(img)
            ax.set_axis_off()

    fig.tight_layout()
    fig.show()
