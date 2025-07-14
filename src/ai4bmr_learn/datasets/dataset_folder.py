from pathlib import Path
from torch.utils.data import Dataset
import tifffile
import torch
from torchvision import tv_tensors
import pandas as pd


class DatasetFolder(Dataset):

    def __init__(self, dataset_dir: Path, image_version: str, split_version: str, transform=None, target_name: str = None):
        super().__init__()

        self.dataset_dir = dataset_dir.resolve()
        assert dataset_dir.exists() and dataset_dir.is_dir()

        self.image_version = image_version
        self.image_dir = self.dataset_dir / "images" / image_version
        assert self.image_dir.exists() and self.image_dir.is_dir()
        self.masks_dir = self.dataset_dir / "masks"
        assert self.masks_dir.exists() and self.masks_dir.is_dir()

        self.image_paths = {i.stem: i for i in self.image_dir.glob("*.tiff")}
        self.mask_paths = {i.stem: i for i in self.masks_dir.glob("*.tiff")}

        # self.metadata_path = self.dataset_dir / "metadata.parquet"
        self.split_path = self.dataset_dir / 'splits' / f"{split_version}.parquet"
        self.metadata = pd.read_parquet(self.split_path)
        # self.sample_ids = sorted(set(self.image_paths) & set(self.mask_paths))
        self.sample_ids = self.metadata.index.tolist()
        # assert set(self.sample_ids) <= set(self.image_paths) & set(self.mask_paths)
        self.targets = self.metadata[target_name] if target_name else None

        self.to_image = lambda x: tv_tensors.Image(torch.tensor(x).long())
        self.to_mask = lambda x: tv_tensors.Mask(torch.tensor(x).double())
        self.transform = transform

    def __getitem__(self, key):
        if isinstance(key, tuple):
            idx, sample_id = key
        else:
            sample_id = self.sample_ids[key]

        image_path = self.image_paths[sample_id]
        image = tifffile.imread(image_path)
        image = self.to_image(image)

        # avoid: RuntimeError: Trying to resize storage that is not resizable
        # mask_path = self.mask_paths[sample_id]
        # mask = tifffile.imread(mask_path)
        # mask = self.to_mask(mask)
        # mask = torch.randn((300, 300)).long()
        # FIXME: including masks triggers: RuntimeError: Trying to resize storage that is not resizable
        # item = {'image': image, 'mask': mask, 'clinical': clinical}

        if self.targets is not None:
            target = self.targets[sample_id]
            item = {'image': image, 'target': target}
        else:
            item = {'image': image}

        if self.transform is not None:
            item = self.transform(item)

        return item

    def __len__(self):
        return len(self.metadata)
