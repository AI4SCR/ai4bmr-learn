import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

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


class MILFromDataset(Dataset):
    def __init__(self, dataset: Dataset,
                 num_instances: int | None = None, shuffle: bool = False, random_state: int | None = None,
                 bag_id_attr: str = 'bag_ids', bag_id_key: str | None = None):

        self.dataset = dataset

        self.bag_id_attr = bag_id_attr
        self.bag_id_key = bag_id_key  # optional key that might be present in the item to assert correctness
        self.bag_ids: list[str|int] | None = None

        self.num_instances = num_instances
        self.shuffle = shuffle
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def setup(self):
        if hasattr(self.dataset, "setup"):
            self.dataset.setup()

        if hasattr(self.dataset, self.bag_id_attr):
            self.bag_ids = sorted(set(getattr(self.dataset, self.bag_id_attr)))
        else:
            raise ValueError(f"Dataset must have {self.bag_id_attr} attribute that identifies the bag ids.")

    def __getitem__(self, idx):
        bag_id = self.bag_ids[idx]
        bag_idc = np.flatnonzero(self.bag_ids == bag_id)

        if self.shuffle:
            self.rng.shuffle(bag_idc)

        num_instances = self.num_instances or len(bag_idc)
        bag_idc = bag_idc[:num_instances]

        instances = []
        for i, idx in enumerate(bag_idc):
            item = self.dataset[idx]
            if self.bag_id_key is not None:
                assert item[self.bag_id_key] == bag_id
            instances.append(item)

        return instances

    def __len__(self):
        return len(self.bag_ids)