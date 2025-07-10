from pathlib import Path
from torch.utils.data import Dataset
import tifffile
import torch
from torchvision.transforms import v2
from torchvision import tv_tensors
import pandas as pd


class DatasetFolder(Dataset):

    def __init__(self, dataset_dir: Path, image_version: str, transform=None):
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
        self.sample_ids = sorted(set(self.image_paths) & set(self.mask_paths))

        self.metadata_path = self.dataset_dir / "metadata.parquet"
        metadata = pd.read_parquet(self.metadata_path)
        self.metadata = metadata

        self.to_image = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=False),
        ])

        self.to_mask = lambda x: tv_tensors.Mask(x).long()

        self.transform = transform

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        image_path = self.image_paths[sample_id]
        image = tifffile.imread(image_path)
        image = self.to_image(image)

        # avoid: RuntimeError: Trying to resize storage that is not resizable
        mask_path = self.mask_paths[sample_id]
        mask = tifffile.imread(mask_path)
        mask = self.to_mask(mask)
        # mask = torch.randn((300, 300)).long()

        metadata = self.metadata.loc[sample_id].fillna('NaN').to_dict()

        # FIXME: including masks triggers: RuntimeError: Trying to resize storage that is not resizable
        # item = {'image': image, 'mask': mask, 'clinical': clinical}
        item = {'image': image, 'metadata': metadata}
        if self.transform is not None:
            item = self.transform(item)

        return item

    def __len__(self):
        return len(self.sample_ids)
