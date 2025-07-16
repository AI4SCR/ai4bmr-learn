
def test_dataset_folder():
    from pathlib import Path
    from ai4bmr_learn.datamodules.dataset_folder import DatasetFolder
    from torch.utils.data import DataLoader
    import ai4bmr_datasets
    from torchvision.transforms import v2
    from ai4bmr_learn.transforms.dino_transform import DINOTransform
    import torch

    train_transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=False),
        DINOTransform(),
    ])

    # TODO: convert to square image first and center crop
    val_transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=False),
        # v2.RandomCrop(224, pad_if_needed=True),
        v2.Resize((224, 224)),
    ])

    dm = self = DatasetFolder(
        dataset=ai4bmr_datasets.Cords2024(),
        train_transform=train_transform,
        val_transform=val_transform,
        target_name='dx_name',
        split_version='clf',
        save_dir=Path('/users/amarti51/prometex/data/dinov1/datasets'),
        force=False,
    )
    dm.prepare_data()
    # dm.generate_splits()
    dm.setup(stage='')
    assert dm.train_set.sample_ids == dm.val_set.sample_ids
    assert dm.train_set.sample_ids == dm.test_set.sample_ids

    idx = 0
    sample_id = dm.train_set.sample_ids[idx]
    key = (idx, sample_id)

    img_idx = dm.val_set[idx]['image']
    img_key = dm.val_set[key]['image']
    assert (img_idx == img_key).all()

    assert dm.train_set[idx]['global_views'][0]['image'].shape[0] == 43
    assert dm.train_set[idx]['global_views'][0]['image'].shape[1] == 224
    assert dm.train_set[idx]['global_views'][0]['image'].shape[2] == 224
    assert dm.val_set[idx]['image'].shape[0] == 43
    assert dm.val_set[idx]['image'].shape[1] == 224
    assert dm.val_set[idx]['image'].shape[2] == 224

    batch = next(iter(DataLoader(dm.train_set, batch_size=4)))
    batch = next(iter(DataLoader(dm.val_set, batch_size=4)))

def test_dataset_folder_with_graph_collator():
    from ai4bmr_learn.collators.graph import Graph
    from pathlib import Path
    from ai4bmr_learn.datamodules.dataset_folder import DatasetFolder
    import ai4bmr_datasets
    from tqdm import tqdm

    stats_path = Path('/users/amarti51/prometex/data/dinov1/datasets/Cords2024/images/default/stats.json')
    cache_dir = Path('/users/amarti51/prometex/data/dinov1/datasets/Cords2024/graphs/radius_32')
    collator = Graph(radius=32, include_self=True, stats_path=stats_path, cache_dir=cache_dir)

    dm = DatasetFolder(
        dataset=ai4bmr_datasets.Cords2024(),
        train_transform=None,
        val_transform=None,
        target_name='dx_name',
        split_version='clf',
        save_dir=Path('/users/amarti51/prometex/data/dinov1/datasets'),
        force=False,
        num_workers=8,
        batch_size=8,
        collate_fn=collator
    )
    dm.setup(stage='')

    _ = [i for i in tqdm(dm.val_dataloader())]
    _ = [i for i in tqdm(dm.train_dataloader())]
    _ = [i for i in tqdm(dm.test_dataloader())]
