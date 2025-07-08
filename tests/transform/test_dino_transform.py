from ai4bmr_learn.transforms.dino_transform import DINOTransform
import numpy as np
from tqdm import tqdm


def test_no_nan_in_rgb_images():
    from ai4bmr_learn.datasets.vocdetection import VOCDetection
    transform = DINOTransform(local_crop_scale=(0.25, 0.4), hf_prob=0, vf_prob=0, rr_prob=0)

    def target_transform(t):
        return 0

    ds = VOCDetection(transform=transform, target_transform=target_transform)

    # check that transform does not yield nan
    for i in tqdm(range(1000)):
        item = ds[i]
        for view in item['global_views'] + item['local_views']:
            img = view['image'].permute(1,2,0).numpy()
            assert not np.isnan(img).any()

def test_no_nan_in_biomedical_images():
    from ai4bmr_datasets import Cords2024
    from torchvision import tv_tensors
    ds = Cords2024()
    ds.setup(image_version='published', mask_version='published')

    transform = DINOTransform(local_crop_scale=(0.25, 0.4), hf_prob=0.25, vf_prob=0.25, rr_prob=0)

    # check that transform does not yield nan
    for i in tqdm(range(100)):

        sample_id = ds.sample_ids[i % len(ds)]
        if sample_id not in ds.images:
            continue
        image = ds.images[sample_id].data
        item = transform({'image': tv_tensors.Image(image)})

        for view in item['global_views'] + item['local_views']:
            img = view['image'].permute(1,2,0).numpy()
            assert not np.isnan(img).any()
