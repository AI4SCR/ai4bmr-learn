
def test_random_crop():
    # %%
    from ai4bmr_learn.transforms.random_crop import RandomCrop
    from ai4bmr_learn.utils.images import get_points
    from ai4bmr_datasets.datasets.DummyPoints import DummyPoints
    from ai4bmr_learn.datasets.PointsPatches import PointsPatches
    from dataclasses import asdict
    from ai4bmr_learn.utils.images import get_coordinates_dict
    from ai4bmr_learn.data_models.Coordinate import PointsCoordinate

    # %%
    height = width = 224
    ds = DummyPoints(height=height, width=width, num_features=16)
    ds.prepare_data(force=False)
    ds.setup()

    # %%
    coord_dicts = get_coordinates_dict(height=ds.height, width=ds.width, kernel_size=32, stride=32)
    coords = [PointsCoordinate(**i, points_path=str(ds.data_dir / f'{sample_id}.gpkg'))
              for sample_id in ds.sample_ids for i in coord_dicts]
    coord = coords[0]

    points = get_points(coord)
    item = dict(
        points=points,
        **asdict(coord)
    )

    # %%
    transform = RandomCrop(scale=(0.05, 1.0))
    for _ in range(1000):
        transform(item)
