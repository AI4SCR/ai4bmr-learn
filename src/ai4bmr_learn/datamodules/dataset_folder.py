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
from torch.utils.data import DataLoader, SubsetRandomSampler
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


class DatasetFolder(L.LightningDataModule):

    def __init__(self,
                 dataset,
                 train_transform: v2.Compose,
                 val_transform: v2.Compose,
                 save_dir: Path,
                 target_name: str,
                 split_version: str,
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

        # SPLIT
        self.split_version = split_version
        self.splits_dir = self.save_dir / 'splits'
        self.splits_path = self.save_dir / 'splits' / f'{self.split_version}.parquet'
        self.metadata_path = self.save_dir / 'metadata.parquet'

        self.train_set = self.val_set = self.test_set = None
        self.train_idc = self.val_idc = self.test_idc = None
        self.train_sampler = self.val_sampler = self.test_sampler = None

        # TRANSFORM
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = None

        # STATS
        self.normalize = "dataset-level"
        self.stats_path = self.images_dir / 'stats.json'
        self.stats_path.parent.mkdir(exist_ok=True, parents=True)

        self.save_hyperparameters(ignore=['train_transform', 'val_transform'])

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

    def generate_splits(self):
        self.generate_ssl_split()
        self.generate_clf_split()

    def generate_ssl_split(self):
        from sklearn.model_selection import train_test_split

        metadata = pd.read_parquet(self.metadata_path, engine='fastparquet')
        sample_ids = metadata.index.tolist()
        metadata = metadata.reset_index()

        # SPLIT
        indices_universe = metadata.index.values

        target_name = self.target_name
        targets = metadata[target_name]

        # select sample ids for which we have metadata
        filter_ = targets.notna().values
        targets = targets[filter_]

        indices = targets.index.values

        # we validate on 'Adenocarcinoma', 'Squamous cell carcinoma' but keep the rest for train
        filter_ = targets.isin(['Adenocarcinoma', 'Squamous cell carcinoma']).values
        targets, indices = targets[filter_], indices[filter_]

        fit_idc, val_idc = train_test_split(indices,
                                            stratify=targets,
                                            test_size=0.2,
                                            random_state=0)

        excl_idc = set(indices_universe.tolist()) - set(fit_idc.tolist()) - set(val_idc.tolist())
        excl_idc = np.array(list(excl_idc))
        fit_idc = np.hstack((fit_idc, excl_idc))
        assert len(fit_idc) + len(val_idc) == len(sample_ids)

        mapping = {'Adenocarcinoma': 0, 'Squamous cell carcinoma': 1}
        metadata.loc[:, self.target_name] = metadata[self.target_name].transform(lambda x: mapping.get(x, -1))

        metadata.loc[fit_idc, 'split'] = 'fit'
        metadata.loc[val_idc, 'split'] = 'val'

        metadata = metadata.set_index('sample_id')

        self.splits_dir.mkdir(parents=True, exist_ok=True)
        splits_path = self.splits_dir / 'ssl.parquet'
        metadata.to_parquet(splits_path)

    def generate_clf_split(self):
        from sklearn.model_selection import train_test_split

        metadata = pd.read_parquet(self.metadata_path, engine='fastparquet')
        metadata = metadata.reset_index()

        # SPLIT
        target_name = self.target_name
        targets = metadata[target_name]

        # we validate on 'Adenocarcinoma', 'Squamous cell carcinoma' but keep the rest for train
        filter_ = targets.notna().values
        filter_ = filter_ & targets.isin(['Adenocarcinoma', 'Squamous cell carcinoma']).values
        metadata = metadata[filter_]
        targets = metadata[target_name]

        indices = targets.index.values
        train_idc, test_idc = train_test_split(indices,
                                               stratify=targets,
                                               test_size=0.2,
                                               random_state=0)

        fit_idc, val_idc = train_test_split(train_idc,
                                            stratify=targets.loc[train_idc],
                                            test_size=0.2,
                                            random_state=0)

        mapping = {'Adenocarcinoma': 0, 'Squamous cell carcinoma': 1}
        metadata.loc[:, self.target_name] = metadata[self.target_name].transform(lambda x: mapping.get(x, -1))

        # sanity checks
        assert set(fit_idc).union(val_idc).union(test_idc) == set(metadata.index.values)
        assert set(fit_idc).intersection(val_idc) == set()
        assert set(val_idc).intersection(test_idc) == set()

        metadata.loc[fit_idc, 'split'] = 'fit'
        metadata.loc[val_idc, 'split'] = 'val'
        metadata.loc[test_idc, 'split'] = 'test'

        metadata = metadata.set_index('sample_id')

        self.splits_dir.mkdir(parents=True, exist_ok=True)
        splits_path = self.splits_dir / 'clf.parquet'
        metadata.to_parquet(splits_path)

    def prepare_data(self) -> None:

        if self.stats_path.exists() and not self.force:
            logger.info(f'Loading stats from {self.stats_path}')
        else:
            logger.info(f'Preparing dataset...')

            ds = self.dataset
            ds.setup(image_version='published', mask_version='published')

            sample_ids = sorted(set(ds.images) & set(ds.masks))
            filter_ = ds.clinical.index.isin(sample_ids)
            ds.clinical[filter_].to_parquet(self.metadata_path, engine='fastparquet')

            self.generate_splits()

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
        from ai4bmr_learn.datasets.dataset_folder import DatasetFolder

        # TRANSFORMS
        normalize = self.get_normalize_transform()
        self.train_transform = v2.Compose([
            self.train_transform,
            normalize
        ])

        self.val_transform = v2.Compose([
            self.val_transform,
            normalize
        ])

        self.train_set = DatasetFolder(
            dataset_dir=self.save_dir,
            image_version=self.image_version,
            split_version=self.split_version,
            transform=self.train_transform,
            target_name=self.target_name
        )

        self.val_set = DatasetFolder(
            dataset_dir=self.save_dir,
            image_version=self.image_version,
            split_version=self.split_version,
            transform=self.val_transform,
            target_name=self.target_name
        )

        self.test_set = DatasetFolder(
            dataset_dir=self.save_dir,
            image_version=self.image_version,
            split_version=self.split_version,
            transform=self.val_transform,
            target_name=self.target_name
        )

        split = pd.read_parquet(self.splits_path)
        split = split.reset_index()

        fit_keys = list(split.loc[split["split"] == "fit", "sample_id"].items())
        val_keys = list(split.loc[split["split"] == "val", "sample_id"].items())
        test_keys = list(split.loc[split["split"] == "test", "sample_id"].items())

        self.train_sampler = SubsetRandomSampler(fit_keys)
        self.val_sampler = SubsetRandomSampler(val_keys)
        self.test_sampler = SubsetRandomSampler(test_keys)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            sampler=self.train_sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            sampler=self.val_sampler,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            sampler=self.test_sampler,
        )
