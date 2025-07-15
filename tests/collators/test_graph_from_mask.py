

def graph():
    from pathlib import Path
    import tifffile
    from torchvision import tv_tensors
    import torch
    from ai4bmr_learn.collators.graph import Graph

    stats_path = Path('/users/amarti51/prometex/data/dinov1/datasets/Cords2024/images/default/stats.json')
    cache_dir = Path('/users/amarti51/prometex/data/dinov1/datasets/Cords2024/graphs/radius_32')
    collator = Graph(radius=32, include_self=True, stats_path=stats_path, cache_dir=cache_dir)
    sample_id = '175_A_1'

    mask_dir = Path('/users/amarti51/prometex/data/dinov1/datasets/Cords2024/masks')
    mask_path = mask_dir / f'{sample_id}.tiff'

    images_dir = Path('/users/amarti51/prometex/data/dinov1/datasets/Cords2024/images/default')
    image_path = images_dir / f'{sample_id}.tiff'

    image = tifffile.imread(image_path)
    image = tv_tensors.Image(torch.tensor(image).float())

    mask = tifffile.imread(mask_path)
    mask = tv_tensors.Mask(torch.tensor(mask).long())

    item = {'image': image, 'mask': mask, 'sample_id': sample_id}
    item = collator.process_item(item)
    assert 'graph' in item

    item1 = {'image': image, 'mask': mask}
    item2 = {'image': image, 'mask': mask}
    batch = collator([item1, item2])
    # assert batch['image'].shape[0] == 2
    # assert batch['mask'].shape[0] == 2
    assert 'graph' in batch