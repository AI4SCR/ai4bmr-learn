# %%
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):

    def __init__(self, *, data: pd.DataFrame, metadata: pd.DataFrame, target_name: str = "target"):
        assert data.index.equals(metadata.index)
        assert metadata.index.duplicated().any() == False
        assert data.isna().any().any() == False

        self.data = data
        self.metadata = metadata
        self.target_name = target_name
        self.targets = metadata[self.target_name]

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