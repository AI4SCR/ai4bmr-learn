# %%
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
from loguru import logger
from torch import get_num_threads
from torch.utils.data import DataLoader

from ..data.splits import Split, generate_splits
from ..datasets.MIL import MILDataset
from ai4bmr_learn.datasets.MIL import MILDataset


@dataclass
class MILDataModuleConfig:
    data_dir: Path
    metadata_path: Path
    target_column_name: str = "target"
    splits_path: Path = None
    test_size: float = 0.2
    val_size: float = 0.0
    num_workers: int = None
    persistent_workers: bool = True
    shuffle: bool = True
    pin_memory: bool = True
    random_state: int = 42


class MILDataModule(L.LightningDataModule):

    def __init__(
        self,
        data_dir: Path,
        metadata_path: Path,
        splits_path: Path = None,
        target_column_name: str = "target",
        test_size: float = 0.2,
        val_size: float = 0.0,
        # batch_size: int = 1,  # note: not supported for MIL
        num_workers: int = None,
        persistent_workers: bool = True,
        shuffle: bool = True,
        pin_memory: bool = True,
        random_state: int = 42,
    ):
        super().__init__()

        # CONFIGURE PATHS
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.splits_path = splits_path or self.metadata_path.parent / "splits.parquet"
        self.target_column_name = target_column_name

        # SPLITTING
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        # DATALOADERS
        self.batch_size = 1
        self.num_workers = num_workers if num_workers is not None else max(0, get_num_threads() - 1)
        self.persistent_workers = persistent_workers if self.num_workers > 0 else False
        self.shuffle = shuffle
        self.pin_memory = pin_memory

        # DATASET
        self.train_idx = self.val_idx = self.test_idx = None
        self.dataset = self.train_set = self.val_set = self.test_set = None

    def setup(self, stage=None):
        from torch.utils.data import Subset

        # NOTE: we use fastparquet because it can preserve cateogrical dtypes for int in contrast to pyarrow
        #   in addition there is an issue with the dtype of the categories after re-loading, WIP
        # metadata = pd.read_parquet(self.metadata_path, engine="fastparquet")
        # metadata = metadata.convert_dtypes()

        splits = pd.read_parquet(self.splits_path, engine="fastparquet")
        splits = splits.convert_dtypes()

        # ATTENTION: IT IS OF HIGHEST IMPORTANCE THAT THE SPLITS AND METADATA ARE ALIGNED
        #   OTHERWISE THE CREATED SUBSET INDICES WILL RETRIVE THE WRONG SAMPLE
        # metadata, splits = metadata.align(splits, axis=0, join="inner")
        # assert metadata.index.equals(splits.index)

        data = {i: pd.read_parquet(self.data_dir / f"{i}.parquet", engine="fastparquet").values for i in splits.index}

        self.dataset = dataset = MILDataset(data=data, metadata=splits, target_column_name=self.target_column_name)

        self.train_idx = np.flatnonzero(splits[Split.COLUMN_NAME] == Split.TRAIN)
        self.val_idx = np.flatnonzero(splits[Split.COLUMN_NAME] == Split.VAL)
        self.test_idx = np.flatnonzero(splits[Split.COLUMN_NAME] == Split.TEST)

        self.train_set = Subset(dataset, self.train_idx)
        self.val_set = Subset(dataset, self.val_idx)
        self.test_set = Subset(dataset, self.test_idx)

    def generate_splits(self, force: bool = False) -> None:
        if not self.splits_path.exists() or force:
            logger.info("Generating splits")
            self.splits_path.parent.mkdir(parents=True, exist_ok=True)
            metadata = pd.read_parquet(self.metadata_path, engine="fastparquet")
            splits = generate_splits(
                metadata,
                target_column_name=self.target_column_name,
                test_size=self.test_size,
                val_size=self.val_size,
                random_state=self.random_state,
            )
            splits.to_parquet(self.splits_path, engine="fastparquet")

    def _prepare_data(self) -> None:
        raise NotImplementedError()

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
