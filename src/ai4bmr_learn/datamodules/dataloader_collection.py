import lightning as L
from torch.utils.data import Dataset
from typing import Callable, Any
import torch.utils.data
from dataclasses import dataclass, asdict
from collections import defaultdict

class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset: Dataset, collate_fn: Any | None = None, **kwargs):
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)


@dataclass
class DataloaderConfig:
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = False
    drop_last: bool = False

    # Advanced hooks
    collate_fn: Callable = None
    sampler: Callable = None


class DataLoaderCollection(L.LightningDataModule):
    VALID_KEYS = {'fit', 'val', 'test', 'predict'}

    def __init__(self, dataloaders: dict[str, list[DataLoader]]):
        super().__init__()
        self.dataloaders = dataloaders

        invalid_keys = set(dataloaders) - self.VALID_KEYS
        assert len(invalid_keys) == 0, f'invalid keys detected: {invalid_keys}. Valid keys are: {self.VALID_KEYS}'

        for k,v in dataloaders.items():
            assert isinstance(v, list), f'Please provide the datasets as lists under each key, i.e fit: [dataset1]'
            assert len(v) == 1, f'Only one dataloader per split supported at this time.'

        # FIX: we need len(ds) to set certain dataloader parameters like `shuffle`

    def setup(self, stage: str | None = None):

        for stage, stage_loaders in self.dataloaders.items():
            for loader in stage_loaders:
                dataset = loader.dataset
                if hasattr(dataset, 'setup'):
                    dataset.setup()

    def train_dataloader(self):
        dl = self.dataloaders.get('fit', None)
        dl = dl[0] if dl is not None else None
        return dl

    def val_dataloader(self):
        dl = self.dataloaders.get('val', None)
        dl = dl[0] if dl is not None else None
        return dl

    def test_dataloader(self):
        dl = self.dataloaders.get('test', None)
        dl = dl[0] if dl is not None else None
        return dl

    def predict_dataloader(self):
        dl = self.dataloaders.get('predict', None)
        dl = dl[0] if dl is not None else None
        return dl

from ai4bmr_learn.utils.utils import to_dict
class DatasetLoaderCollection(L.LightningDataModule):
    VALID_KEYS = {'fit', 'val', 'test', 'predict'}

    def __init__(self, datasets: dict[str, list[Dataset]], dataloaders: dict[str, list[DataloaderConfig]]):
        super().__init__()
        assert set(datasets) == set(dataloaders), f'Keys in datasets and dataloaders to not match'

        invalid_keys = set(dataloaders) - self.VALID_KEYS
        assert len(invalid_keys) == 0, f'invalid keys detected: {invalid_keys}. Valid keys are: {self.VALID_KEYS}'

        invalid_keys = set(datasets) - self.VALID_KEYS
        assert len(invalid_keys) == 0, f'invalid keys detected: {invalid_keys}. Valid keys are: {self.VALID_KEYS}'

        for k,v in dataloaders.items():
            assert isinstance(v, list), f'Please provide the datasets as lists under each key, i.e fit: [dataset1]'
            assert len(v) == 1, f'Only one dataloader per split supported at this time.'

        for k,v in datasets.items():
            assert isinstance(v, list), f'Please provide the datasets as lists under each key, i.e fit: [dataset1]'
            assert len(v) == 1, f'Only one dataloader per split supported at this time.'

        for k,v in datasets.items():
            loaders = dataloaders[k]
            assert len(v) == len(loaders), f'Number of datasets and number of dataloaders does not match'

        self.datasets = datasets
        self.dataloader_configs = dataloaders
        self.dataloaders = defaultdict(list)

        self.save_hyperparameters()

    def setup(self, stage: str | None = None):

        for stage, datasets in self.datasets.items():
            for dataset in datasets:
                if hasattr(dataset, 'setup'):
                    dataset.setup()

        for stage in self.VALID_KEYS:
            if stage not in self.datasets:
                continue
            datasets = self.datasets[stage]
            configs = self.dataloader_configs[stage]
            assert len(datasets) == len(configs), f'Number of datasets and number of dataloaders does not match'
            for dataset, config in zip(datasets, configs):
                config = asdict(config)
                loader = torch.utils.data.DataLoader(dataset=dataset, **config)
                self.dataloaders[stage].append(loader)

    def train_dataloader(self):
        dl = self.dataloaders.get('fit', None)
        dl = dl[0] if dl is not None else []
        return dl

    def val_dataloader(self):
        dl = self.dataloaders.get('val', None)
        dl = dl[0] if dl is not None else []
        return dl

    def test_dataloader(self):
        dl = self.dataloaders.get('test', None)
        dl = dl[0] if dl is not None else []
        return dl

    def predict_dataloader(self):
        dl = self.dataloaders.get('predict', None)
        dl = dl[0] if dl is not None else []
        return dl