# %%
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MILDataset(Dataset):

    def __init__(self, *, data: dict, metadata: pd.DataFrame, target_column_name: str = "target"):
        assert metadata.index.duplicated().any() == False
        assert target_column_name in metadata.columns

        self.data = data
        self.metadata = metadata
        self.target_column_name = target_column_name
        self.targets = metadata[self.target_column_name]

        assert self.targets.isna().any().any() == False

        self.sample_ids = sorted(self.metadata.index)
        assert set(self.sample_ids) <= set(self.data.keys())

        self.num_samples = len(data)
        self.num_features = self.data[self.sample_ids[0]].shape[1]

        if self.targets.dtype == "category":
            self.is_categorical = True
            self.label_to_index = {k: v for k, v in zip(self.targets.cat.categories, self.targets.cat.codes)}
            self.labels = self.targets.cat.categories.to_list()
            # NOTE: we use the codes to ensure that the targets are integers
            self.targets = self.targets.cat.codes
            self.num_classes = len(self.labels)
            self.class_distribution = torch.tensor(np.bincount(self.targets))
        else:
            self.is_categorical = False
            self.labels = self.num_classes = self.label_to_index = self.class_distribution = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        # note: cloned to avoid RuntimeError: Trying to resize storage that is not resizable
        x = torch.tensor(self.data[sample_id]).float()
        target = torch.tensor(self.targets.loc[sample_id])
        target = target.long() if self.is_categorical else target.float()
        metadata = self.metadata.loc[sample_id].to_dict()
        return dict(x=x, target=target, metadata=metadata)
