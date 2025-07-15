from ai4bmr_learn.datasets.dataset_folder import DatasetFolder
from pathlib import Path
dataset_dir = Path('/users/amarti51/prometex/data/dinov1/datasets/Cords2024')
image_version = 'default'
split_version = 'clf'
ds = DatasetFolder(dataset_dir=dataset_dir, image_version=image_version, split_version=split_version)
ds[0]
