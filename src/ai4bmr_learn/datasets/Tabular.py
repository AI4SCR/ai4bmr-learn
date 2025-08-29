# %%
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
from pathlib import Path

import ai4bmr_learn.utils.utils
import json
from copy import deepcopy
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset
from torchvision import tv_tensors

from ai4bmr_learn.datasets.utils import filter_items_and_metadata
from ai4bmr_learn.utils import io
from ai4bmr_learn.utils.utils import pair
from torchvision.transforms import v2
from PIL.Image import Image

class TabularDataset(Dataset):

    def __init__(self, *, data: pd.DataFrame, metadata: pd.DataFrame, target_column_name: str = "target"):
        assert data.index.equals(metadata.index)
        assert metadata.index.duplicated().any() == False
        assert data.isna().any().any() == False
        assert target_column_name not in data.columns
        assert target_column_name in metadata.columns

        self.data = data
        self.metadata = metadata
        self.target_column_name = target_column_name
        self.targets = metadata[self.target_column_name]

        assert self.targets.isna().any().any() == False

        self.num_samples = data.shape[0]
        self.num_features = data.shape[1]

        if self.targets.dtype == "category":
            self.is_categorical = True
            self.label_encoder = LabelEncoder()

            self.labels = self.targets.cat.categories.to_list()
            self.label_encoder.fit(self.labels)
            self.label_to_target = {k:v for k, v in zip(self.labels, self.label_encoder.transform(self.labels))}

            # NOTE: we use the codes to ensure that the targets are integers, self.targets.cat.codes
            self.targets = pd.Series(self.label_encoder.transform(self.targets), index=self.targets.index)
            self.num_classes = len(self.labels)
            self.class_distribution = torch.tensor(np.bincount(self.targets), dtype=torch.long)
        else:
            self.is_categorical = False
            self.label_encoder = self.labels = self.num_classes = self.label_to_index = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.tensor(self.data.iloc[idx].values).float()
        target = torch.tensor(self.targets.iloc[idx])
        target = target.long() if self.is_categorical else target.float()
        metadata = ai4bmr_learn.utils.utils.to_dict()
        return dict(x=x, target=target, metadata=metadata)

    @classmethod
    def from_paths(cls, data_path: Path, metadata_path: Path, target_column_name: str = "target") -> "TabularDataset":
        data = pd.read_parquet(data_path, engine="fastparquet")
        metadata = pd.read_parquet(metadata_path, engine="fastparquet")
        return cls(data=data, metadata=metadata, target_column_name=target_column_name)


class Tabular(Dataset):
    name: str = "Tabular"

    def __init__(self, *,
                 data_path: Path,
                 metadata_path: Path,
                 split: str | None = None,
                 transform: Callable | None = None,
                 cache_dir: Path | None = None,
                 drop_nan_columns: bool = False,
                 id_key: str | None = None,
                 ):

        if id_key is not None:
            raise ValueError('In a Tabular dataset the id_key must be the index.')

        # TODO: use variable and labels name to distinguish regression from classification
        assert data_path.exists()
        self.data_path = data_path.resolve()
        self.data: pd.DataFrame | None = None

        assert metadata_path.exists()
        self.metadata_path = metadata_path.resolve()
        self.metadata: pd.DataFrame | None = None
        self.item_ids: list[str | int | tuple] | None = None

        self.num_samples: int | None = None
        self.num_features: int | None = None

        # METADATA
        self.metadata_path = metadata_path
        if metadata_path is not None:
            self.metadata_path = Path(metadata_path).expanduser().resolve()
            assert self.metadata_path.exists(), f'metadata_path {self.metadata_path} does not exist'
        self.metadata: pd.DataFrame | None = None
        self.drop_nan_columns = drop_nan_columns
        self.split = split

        # CACHE
        self.cache_dir = cache_dir.resolve() if cache_dir else None
        self.cached_ids: list[str] | None = None

        # TRANSFORM
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def sanity_checks(self):
        assert self.data.index.equals(self.metadata.index)
        assert self.metadata.index.duplicated().any() == False
        assert self.data.isna().any().any() == False

    def setup(self):
        logger.info(f'Setting up {self.name} dataset from data_path: {self.data_path} and metadata_path: {self.metadata_path}')

        data = pd.read_parquet(self.data_path, engine="fastparquet")
        logger.info(f'Loaded data with shape {data.shape}')

        metadata = pd.read_parquet(self.metadata_path, engine="fastparquet")
        assert set(metadata.index) >= set(data.index)

        item_ids = data.index.tolist()
        self.item_ids, self.metadata = filter_items_and_metadata(item_ids=item_ids,
                                                                 metadata=metadata,
                                                                 split=self.split,
                                                                 drop_nan_columns=self.drop_nan_columns)
        self.data = data.loc[self.item_ids]
        assert len(self.data) == len(self.item_ids)

        self.num_samples = self.data.shape[0]
        self.num_features = self.data.shape[1]
        self.sanity_checks()

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        sample_id = item.name

        data = torch.tensor(item.values).float()

        metadata = self.metadata.loc[sample_id]
        assert metadata.isna().any().any() == False
        metadata = metadata.to_dict()

        return dict(sample_id=sample_id, data=data, metadata=metadata)