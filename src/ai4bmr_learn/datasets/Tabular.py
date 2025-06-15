# %%
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
from pathlib import Path

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
        metadata = self.metadata.iloc[idx].to_dict()
        return dict(x=x, target=target, metadata=metadata)

    @classmethod
    def from_paths(cls, data_path: Path, metadata_path: Path, target_column_name: str = "target") -> "TabularDataset":
        data = pd.read_parquet(data_path, engine="fastparquet")
        metadata = pd.read_parquet(metadata_path, engine="fastparquet")
        return cls(data=data, metadata=metadata, target_column_name=target_column_name)