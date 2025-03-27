# %%
from pathlib import Path
from torch import get_num_threads
from ai4bmr_learn.data.splits import Split, generate_splits
import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TabularDataset(Dataset):

    def __init__(self, data: pd.DataFrame, metadata: pd.DataFrame, target_name: str = "target"):
        super().__init__()

        assert data.index.equals(metadata.index)
        assert metadata.index.duplicated().any() == False
        assert data.isna().any().any() == False

        self.data = data
        self.metadata = metadata
        self.targets = metadata[target_name]

        assert self.targets.isna().any().any() == False

        self.num_samples = data.shape[0]
        self.num_features = data.shape[1]

        if self.targets.dtype == "category":
            self.is_categorical = True
            self.classes = self.targets.cat.categories
            self.num_classes = len(self.classes)
            self.class_name_to_index = {k: v for k, v in zip(self.targets.cat.categories, self.targets.cat.codes)}
        else:
            self.is_categorical = False
            self.classes = self.num_classes = self.class_name_to_index = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data.iloc[idx]).float()
        target = torch.tensor(self.targets[idx])
        target = target.long() if self.is_categorical else target.float()
        metadata = self.metadata.iloc[idx].to_dict()
        return dict(x=x, target=target, metadata=metadata)


class TabularDataModule(L.LightningDataModule):

    def __init__(
        self,
        data_path: Path,
        metadata_path: Path,
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
