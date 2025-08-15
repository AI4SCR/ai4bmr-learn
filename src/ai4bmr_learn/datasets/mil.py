import logging
from typing import Callable

import glom
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from tqdm import tqdm

import ai4bmr_learn.utils.utils


class MIL(Dataset):

    def __init__(self, data_dir: Path, metadata_path: Path, bag_id_col: str, num_instances: int):
        assert metadata_path.exists()
        self.metadata_path = metadata_path.resolve()
        self.bag_id_col = bag_id_col
        self.num_instances = num_instances

    def setup(self):
        self.metadata = pd.read_parquet(self.metadata_path)
        self.bag_ids = sorted(set(self.metadata[self.bag_id_col]))

    def __len__(self):
        return len(self.bag_ids)

    def __getitem__(self, idx):
        pass

from torch.utils.data._utils.collate import collate, default_collate_fn_map
import pickle

class MILFromDataset(Dataset):
    def __init__(self, dataset: Dataset, collator: Callable | None | bool = None,
                 num_instances: int | None = None, pad: bool = False, attention_key: str = 'attention',
                 shuffle: bool = False, random_state: int | None = None,
                 bag_ids_attr: str = 'bag_ids', bag_id_key: str | None = None, cache_path: Path | None = None,
                 progress_bar: bool = False):

        self.dataset = dataset
        if collator is None and not False:
            self.collator = lambda x: collate(x, collate_fn_map=default_collate_fn_map)
        else:
            self.collator = collator

        self.bag_id_attr = bag_ids_attr
        self.bag_id_key = bag_id_key  # optional key that might be present in the item to assert correctness
        self.dataset_bag_ids: list[str | int] | None = None
        self.bag_ids: list[str | int] | None = None
        self.cache_dir = cache_path

        self.num_instances = num_instances
        self.pad = pad
        self.attention_key = attention_key
        self.shuffle = shuffle
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.progress_bar = progress_bar

        if self.num_instances is not None and self.shuffle and self.cache_dir is not None:
            logging.warning(f'Using cache with `shuffle=True` and num_instance=True`. A random subset of instances will be cached once.')

    def setup(self):
        if hasattr(self.dataset, "setup"):
            self.dataset.setup()

        if hasattr(self.dataset, self.bag_id_attr):
            self.dataset_bag_ids = getattr(self.dataset, self.bag_id_attr)
            self.bag_ids = sorted(set(self.dataset_bag_ids))
        else:
            raise ValueError(f"Dataset must have {self.bag_id_attr} attribute that identifies the bag ids.")

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_bag(self, bag, id: str):
        with open(self.cache_dir / f"{id}.pkl", "wb") as f:
            pickle.dump(bag, f)

    def load_bag(self, id: str):
        with open(self.cache_dir / f"{id}.pkl", "rb") as f:
            bag = pickle.load(f)
        return bag

    def has_cache(self, id: str):
        return (self.cache_dir / f"{id}.pkl").exists()

    def invalidate_cache(self):
        import shutil
        shutil.rmtree(self.cache_dir)

    def __getitem__(self, idx):
        bag_id = self.bag_ids[idx]

        if self.cache_dir is not None and self.has_cache(bag_id):
            return self.load_bag(bag_id)

        bag_idc = np.flatnonzero(np.array(self.dataset_bag_ids) == bag_id)  # FIX: this needs to be the attr

        if self.shuffle:
            self.rng.shuffle(bag_idc)

        num_instances = self.num_instances or len(bag_idc)
        bag_idc = bag_idc[:num_instances]

        bag = []
        iterator = tqdm(bag_idc) if self.progress_bar else bag_idc
        for idx in iterator:
            item = self.dataset[idx]
            # NOTE: we do not need attention if we do not pad
            # item = glom.assign(item, self.attention_key, True, missing=lambda: {})
            if self.bag_id_key is not None:
                assert item[self.bag_id_key] == bag_id
            bag.append(item)

        if self.pad and self.num_instances is not None:
            num_pad = self.num_instances - len(bag_idc)
            for _ in range(num_pad):
                item = self.dataset[bag_idc[-1]]
                item = glom.assign(item, self.attention_key, False, missing=lambda: {})
                bag.append(item)

        if self.collator is not False:
            bag = self.collator(bag)

        if isinstance(bag, dict):
            bag['bag_id'] = bag_id

        if self.cache_dir is not None:
            self.cache_bag(bag, bag_id)

        return bag

    def __len__(self):
        return len(self.bag_ids)


class MILDataset(Dataset):

    def __init__(self, *, data: dict, metadata: pd.DataFrame, target_column_name: str = "target"):
        assert metadata.index.duplicated().any() == False
        assert target_column_name in metadata.columns

        self.data = data
        self.metadata = metadata
        self.target_column_name = target_column_name
        self.targets = metadata[self.target_column_name]

        assert self.targets.isna().any().any() == False

        self.sample_ids = self.metadata.index
        assert set(self.sample_ids) <= set(self.data.keys())

        self.num_samples = len(data)
        self.num_features = self.data[self.sample_ids[0]].shape[1]

        if self.targets.dtype == "category":
            self.is_categorical = True
            self.label_encoder = LabelEncoder()

            self.labels = self.targets.cat.categories.to_list()
            self.label_encoder.fit(self.labels)
            self.label_to_target = {k: v for k, v in zip(self.labels, self.label_encoder.transform(self.labels))}

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
        sample_id = self.metadata.index[idx]
        # note: don't use from_numpy to avoid RuntimeError: Trying to resize storage that is not resizable
        x = torch.tensor(self.data[sample_id], dtype=torch.float)

        dtype = torch.long if self.is_categorical else torch.float
        target = torch.tensor(self.targets.loc[sample_id], dtype=dtype)

        metadata = ai4bmr_learn.utils.utils.to_dict()
        return dict(x=x, target=target, metadata=metadata, sample_id=sample_id)
