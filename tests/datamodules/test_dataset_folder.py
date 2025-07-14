
def test_dataset_folder():
    from pathlib import Path
    from ai4bmr_learn.datamodules.dataset_folder import DatasetFolder
    from torch.utils.data import DataLoader
    import ai4bmr_datasets

    dm = self = DatasetFolder(
        dataset=ai4bmr_datasets.Cords2024(),
        target_name='dx_name',
        save_dir=Path('/users/amarti51/prometex/data/dinov1/datasets'),
        force=False,
    )
    dm.prepare_data()
    dm.setup(stage='')
    assert dm.train_set[0]['global_views'][0]['image'].shape[0] == 43
    assert dm.val_set[0]['image'].shape[0] == 43

    batch = next(iter(DataLoader(dm.train_set, batch_size=4, shuffle=True)))
    batch = next(iter(DataLoader(dm.val_set, batch_size=4, shuffle=True)))
