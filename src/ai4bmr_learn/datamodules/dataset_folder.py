import json
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from ai4bmr_core.utils.saving import save_image, save_mask
from ai4bmr_core.utils.stats import StatsRecorder
from loguru import logger
from torch import get_num_threads
from torch.utils.data import DataLoader
from torchvision.transforms import v2
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


class DatasetFolder(L.LightningDataModule):

    def __init__(self,
                 dataset,
                 save_dir: Path,
                 target_name: str,
                 force: bool = False,
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
        self.force = force
        self.dataset = dataset
        self.target_name = target_name
        self.save_dir = save_dir.resolve() / dataset.name

        self.image_version = 'default'
        self.images_dir = self.save_dir / 'images' / self.image_version
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir = self.save_dir / 'masks'
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        self.set = None
        self.train_idc = self.val_idc = self.test_idc = None
        self.train_sampler = self.val_sampler = self.test_sampler = None
        self.train_transform = self.val_transform = self.test_transform = None

        # STATS
        self.normalize = "dataset-level"
        self.stats_path = self.images_dir / 'stats.json'
        self.stats_path.parent.mkdir(exist_ok=True, parents=True)

        self.save_hyperparameters()

    def get_normalize_transform(self):
        if self.normalize == "dataset-level":
            assert self.stats_path.exists(), f"No dataset level stats found at: {self.stats_path}"
            with open(self.stats_path, "r") as f:
                stats = json.load(f)
                mean, std = stats['mean'], stats['std']
                return v2.Normalize(mean=mean, std=std)
        elif self.normalize == "sample-level":
            raise NotImplementedError()
        else:
            return v2.Identity()

    def prepare_data(self) -> None:
        if self.stats_path.exists() and not self.force:
            logger.info(f'Loading stats from {self.stats_path}')
        else:
            logger.info(f'Preparing dataset...')

            ds = self.dataset
            ds.setup(image_version='published', mask_version='published')
            ds.clinical.to_parquet(self.save_dir / 'metadata.parquet', engine='fastparquet')

            sample_ids = sorted(set(ds.images) & set(ds.masks))

            sr = StatsRecorder()
            for i, sample_id in enumerate(sample_ids, start=1):
                logger.info(f"Processing {i}/{len(sample_ids)}")

                mask = ds.masks[sample_id].data
                image = ds.images[sample_id].data
                image = normalize(image)
                sr.update(image)

                save_image(image, self.images_dir / f"{sample_id}.tiff")
                save_mask(mask, self.masks_dir / f"{sample_id}.tiff")

            pd.Series(sr.__dict__).to_json(self.stats_path)

    def setup(self, stage):
        from sklearn.model_selection import train_test_split
        from torch.utils.data import SubsetRandomSampler
        from ai4bmr_learn.datasets.dataset_folder import DatasetFolder

        # TRANSFORMS
        normalize = self.get_normalize_transform()
        self.train_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=False),
            DINOTransform(),
            normalize,
        ])

        # TODO: convert to square image first and center crop
        self.val_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=False),
            # v2.RandomCrop(224, pad_if_needed=True),
            v2.Resize((224, 224)),
            normalize
        ])

        self.set = ds = DatasetFolder(dataset_dir=self.save_dir,
                           image_version=self.image_version,
                           transform=None)

        # SPLIT
        indices_universe = torch.tensor(range(len(ds.sample_ids)))

        target_name = self.target_name
        metadata = ds.metadata
        targets = metadata[target_name]

        # select sample ids for which we have metadata
        filter_ = targets.index.isin(ds.sample_ids)
        filter_ = filter_ & targets.notna().values
        targets = targets[filter_]

        indices = torch.tensor([ds.sample_ids.index(i) for i in targets.index])

        # we validate on 'Adenocarcinoma', 'Squamous cell carcinoma' but keep the rest for train
        filter_ = targets.isin(['Adenocarcinoma', 'Squamous cell carcinoma']).values
        targets, indices = targets[filter_], indices[filter_]

        train_idc, self.val_idc = train_test_split(indices,
                                                   stratify=targets,
                                                   test_size=0.2,
                                                   random_state=0)

        excl_idc = set(indices_universe.tolist()) - set(train_idc.tolist()) - set(self.val_idc.tolist())
        excl_idc = torch.tensor(list(excl_idc), dtype=indices_universe.dtype)
        self.train_idc = torch.hstack((train_idc, excl_idc))
        assert len(self.train_idc) + len(self.val_idc) == len(ds.sample_ids)

        self.train_sampler = SubsetRandomSampler(self.train_idc)
        self.val_sampler = SubsetRandomSampler(self.val_idc)

    def train_dataloader(self):
        return DataLoader(
            self.set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            sampler=self.train_sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            sampler=self.val_sampler,
        )


# import ai4bmr_datasets
#
# dm = self = DatasetFolder(
#     dataset=ai4bmr_datasets.Cords2024(),
#     target_name='dx_name',
#     save_dir=Path('/users/amarti51/prometex/data/dinov1/datasets'),
#     force=False,
# )
# dm.prepare_data()
# dm.setup(stage='')
# dm.set[0]
