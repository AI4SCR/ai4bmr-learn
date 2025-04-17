# %%

import pandas as pd
import torch
from torch.utils.data import Dataset


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
            self.label_to_index = {k: v for k, v in zip(self.targets.cat.categories, self.targets.cat.codes)}
            self.labels = self.targets.cat.categories.to_list()
            # NOTE: we use the codes to ensure that the targets are integers
            self.targets = self.targets.cat.codes
            self.num_classes = len(self.labels)
            self.class_distribution = torch.tensor(np.bincount(self.targets), dtype=torch.long)
        else:
            self.is_categorical = False
            self.labels = self.num_classes = self.label_to_index = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.tensor(self.data.iloc[idx].values).float()
        target = torch.tensor(self.targets.iloc[idx])
        target = target.long() if self.is_categorical else target.float()
        metadata = self.metadata.iloc[idx].to_dict()
        return dict(x=x, target=target, metadata=metadata)
