from ai4bmr_learn.datasets.dataset_folder import DatasetFolder
from pathlib import Path
dataset_dir = Path('/users/amarti51/prometex/data/benchmarking/datasets/Cords2024')

image_version = 'published'
mask_version = 'published'
split_version = 'clf-target=dx_name'
annotation_version = 'cell_type'
ds = DatasetFolder(dataset_dir=dataset_dir, image_version=image_version, mask_version=mask_version, split_version=split_version, annotation_version=annotation_version)
ds[0]
item = ds[0]

image_version = 'published'
split_version = 'clf-target=dx_name'
ds = DatasetFolder(dataset_dir=dataset_dir, image_version=image_version, split_version=split_version)
ds[0]
