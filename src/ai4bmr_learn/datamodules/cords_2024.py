from pathlib import Path
import lightning as L
import numpy as np
import torch
from loguru import logger
from torch import get_num_threads
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2
from torchvision import tv_tensors
from torch.utils.data import Dataset

import pickle

from ai4bmr_core.utils.stats import StatsRecorder
from ai4bmr_learn.transforms.dino_transform import DINOTransform


# %% HELPER
def normalize(img, censoring=0.999, cofactor=1, exclude_zeros=True):
    img = np.arcsinh(img / cofactor)

    if exclude_zeros:
        masked_img = np.where(img == 0, np.nan, img)
        thres = np.nanquantile(masked_img, censoring, axis=(1, 2), keepdims=True)
    else:
        thres = np.quantile(img, q=censoring, axis=(1, 2), keepdims=True)

    img = np.minimum(img, thres)

    return img


class Wrapper(Dataset):

    def __init__(self, base_dir: Path, transform=None):
        from ai4bmr_datasets.datasets.Cords2024 import Cords2024
        super().__init__()

        self.ds = Cords2024(base_dir=base_dir)
        self.ds.setup(image_version='published', mask_version='published')
        self.ds.sample_ids = sorted(self.ds.sample_ids)

        self.transform = transform

    def __getitem__(self, idx):
        sample_id = self.ds.sample_ids[idx]

        image = self.ds.images[sample_id].data
        image = normalize(image)
        image = tv_tensors.Image(image)

        mask = self.ds.masks[sample_id].data
        mask = tv_tensors.Mask(mask)

        clinical = self.ds.clinical.loc[sample_id].dropna().to_dict()

        item = {'image': image, 'mask': mask, 'clinical': clinical}
        if self.transform is not None:
            item = self.transform(item)

        return item

    def __len__(self):
        return len(self.ds)


#
class Cords2024(L.LightningDataModule):

    def __init__(self,
                 batch_size: int = 64,
                 num_workers: int = None,
                 persistent_workers: bool = True,
                 shuffle: bool = True,
                 pin_memory: bool = True
                 ):
        super().__init__()

        # DATALOADERS
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else max(0, get_num_threads() - 1)
        self.persistent_workers = persistent_workers if self.num_workers > 0 else False
        self.shuffle = shuffle
        self.pin_memory = pin_memory

        # DATASET
        self.base_dir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/Cords2024')
        self.base_dir = self.base_dir.resolve()
        assert self.base_dir.exists()

        self.train_idc = self.val_idc = self.test_idc = None
        self.train_set = self.val_set = self.test_set = None

        # STATS
        self.normalize = "dataset-level"
        self.stats_path = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/dinov1/datasets/Cords2024/stats.pkl')
        self.stats_path.parent.mkdir(exist_ok=True, parents=True)

        self.save_hyperparameters()

    def get_normalize_transform(self):
        if self.normalize == "dataset-level":
            assert self.stats_path.exists(), f"No dataset level stats found at: {self.stats_path}"
            with open(self.stats_path, "rb") as f:
                sr = pickle.load(f)
                return v2.Normalize(mean=sr.mean, std=sr.std)
        elif self.normalize == "sample-level":
            raise NotImplementedError()
        else:
            return v2.Identity()

    def prepare_data(self) -> None:
        ds = Wrapper(base_dir=self.base_dir, transform=None)
        sr = StatsRecorder()
        for i, item in enumerate(ds, start=1):
            logger.info(f"Processing {i}/{len(ds)}")
            sr.update(item['image'].numpy())

        with open(self.stats_path, 'wb') as f:
            pickle.dump(sr, f)

    def setup(self, stage):
        from sklearn.model_selection import train_test_split

        # TRANSFORMS
        normalize = self.get_normalize_transform()
        train_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=False),
            DINOTransform(),
            normalize,
        ])

        val_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=False),
            v2.RandomCrop(224, pad_if_needed=True),
            # v2.Resize((224, 224)),
            normalize
        ])

        ds_train = Wrapper(base_dir=self.base_dir, transform=train_transform)
        ds_val = Wrapper(base_dir=self.base_dir, transform=val_transform)

        # SPLIT
        metadata = ds_train.ds.clinical.loc[ds_train.ds.sample_ids]
        # we validate on 'Adenocarcinoma', 'Squamous cell carcinoma' but keep the rest for train
        filter_ = metadata.dx_name.isin(['Adenocarcinoma', 'Squamous cell carcinoma']).values

        indices = torch.arange(len(ds_train))
        train_idc_1 = indices[~filter_]

        metadata = metadata[filter_]
        train_idc_2, val_idc = train_test_split(indices[filter_], stratify=metadata['dx_name'],
                                                test_size=0.2, random_state=0)
        train_idc = torch.hstack((train_idc_1, train_idc_2))

        self.train_set = Subset(ds_train, indices=train_idc)
        self.val_set = Subset(ds_val, indices=val_idc)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )


dm = self = Cords2024()
# dm.prepare_data()
# dm.setup(stage=None)
# dm.train_set[0]
