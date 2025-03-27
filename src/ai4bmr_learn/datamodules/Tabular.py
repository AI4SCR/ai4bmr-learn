# %%
from pathlib import Path
from torch import get_num_threads
from ..data.splits import Split, generate_splits
import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class DummyDataModule(L.LightningDataModule):

    def __init__(
        self,
        data_path: Path = Path('~/data/ai4bmr-learn/DummyTabular/data.parquet').expanduser().resolve(),
        metadata_path: Path = Path('~/data/ai4bmr-learn/DummyTabular/metadata.parquet').expanduser().resolve(),
        splits_path: Path = None,
        test_size: float = 0.2,
        val_size: float = 0.0,
        batch_size: int = 64,
        num_workers: int = None,
        persistent_workers: bool = True,
        shuffle: bool = True,
        pin_memory: bool = True,
        random_state: int = 42,
    ):
        super().__init__()

        # CONFIGURE PATHS
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.splits_path = splits_path or self.metadata_path.parent / "splits.parquet"

        # SPLITTING
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        # DATALOADERS
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else max(0, get_num_threads() - 1)
        self.persistent_workers = persistent_workers if self.num_workers > 0 else False
        self.shuffle = shuffle
        self.pin_memory = pin_memory

        # DATASET
        self.train_idx = self.val_idx = self.test_idx = None
        self.train_set = self.val_set = self.test_set = None

    def setup(self, stage=None):
        from torch.utils.data import Subset
        from ..datasets.Tabular import TabularDataset

        data = pd.read_parquet(self.data_path)
        metadata = pd.read_parquet(self.metadata_path)
        splits = pd.read_parquet(self.splits_path)

        dataset = TabularDataset(data=data, metadata=metadata)

        self.train_idx = np.flatnonzero(splits[Split.COLUMN_NAME] == Split.TRAIN)
        self.val_idx = np.flatnonzero(splits[Split.COLUMN_NAME] == Split.VAL)
        self.test_idx = np.flatnonzero(splits[Split.COLUMN_NAME] == Split.TEST)

        self.train_set = Subset(dataset, self.train_idx)
        self.val_set = Subset(dataset, self.val_idx)
        self.test_set = Subset(dataset, self.test_idx)

    def generate_splits(self):
        if not self.splits_path.exists():
            self.splits_path.parent.mkdir(parents=True, exist_ok=True)
            metadata = pd.read_parquet(self.metadata_path)
            splits = generate_splits(
                metadata, test_size=self.test_size, val_size=self.val_size, random_state=self.random_state
            )
            splits.to_parquet(self.splits_path)

    def _prepare_data(self) -> None:
        # NOTE: here we load one of our datasets and bring it into the right format for the training that we want to do.
        from ai4bmr_datasets.datasets.DummyTabular import DummyTabular

        ds = DummyTabular(num_samples=1000, num_classes=2, num_features=10)
        data = ds.load()

        if not self.data_path.exists():
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            data['data'].to_parquet(self.data_path)

        if not self.metadata_path.exists():
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            data['metadata'].to_parquet(self.metadata_path)

    def prepare_data(self) -> None:
        self._prepare_data()
        self.generate_splits()

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

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )
