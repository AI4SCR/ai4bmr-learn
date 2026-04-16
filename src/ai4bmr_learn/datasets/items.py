import json
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset

from ai4bmr_learn.datasets.utils import filter_items_and_metadata


class Items(Dataset):
    name: str = 'Items'

    def __init__(
            self,
            items_path: Path,
            metadata_path: Path | None = None,
            split: str | None = None,
            transform: Callable | None = None,
            cache_dir: Path | None = None,
            drop_nan_columns: bool = False,
            id_key: str | None = None,
            num_workers: int = 10,
            batch_size: int = 32,
    ):
        super().__init__()

        self.items_path = Path(items_path).expanduser().resolve()
        assert self.items_path.exists(), f'items_path {self.items_path} does not exist'
        self.items: list[dict] | None = None
        self.id_key = id_key
        self.item_ids: list[str | int] | None = None

        self.metadata_path = metadata_path
        if metadata_path is not None:
            assert id_key is not None, 'id_key must be provided when metadata_path is provided'
            self.metadata_path = Path(metadata_path).expanduser().resolve()
            assert self.metadata_path.exists(), f'metadata_path {self.metadata_path} does not exist'
        self.metadata: pd.DataFrame | None = None
        self.drop_nan_columns = drop_nan_columns
        self.split = split

        self.cache_dir = cache_dir.resolve() if cache_dir else None
        self.cached_ids: list[str] | None = None
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx) -> dict:
        raise NotImplementedError('Inherit from `Items` to create your own dataset subclass.')

    def setup(self):
        logger.info(f'Setting up {self.name} dataset from items_path: {self.items_path}')

        with open(self.items_path, 'r') as f:
            self.items = json.load(f)
            self.item_ids = [i[self.id_key] for i in self.items] if self.id_key else None
            logger.info(f'Loaded {len(self.items)} items')

        if self.metadata_path is not None:
            logger.info(f'Loading metadata from {self.metadata_path}')
            item_ids = [i[self.id_key] for i in self.items]
            metadata = pd.read_parquet(self.metadata_path)
            self.item_ids, self.metadata = filter_items_and_metadata(
                item_ids=item_ids,
                metadata=metadata,
                split=self.split,
                drop_nan_columns=self.drop_nan_columns,
            )
            item_id_set = set(self.item_ids)
            self.items = [i for i in self.items if i[self.id_key] in item_id_set]

        if self.cache_dir and not self.has_cache():
            logger.info(f'No cache found at {self.cache_dir}. Creating...')
            self.create_cache()

    def has_cache(self, iid: str | None = None) -> bool:
        if self.cache_dir is None:
            return False

        if self.cached_ids is None:
            logger.info('Gather cached ids...')
            self.cached_ids = set([i.stem for i in self.cache_dir.rglob('*.pt')])

        if iid is not None:
            return iid in self.cached_ids

        iid = set([item[self.id_key] for item in self.items])
        if iid <= self.cached_ids:
            logger.info(f'Found all {len(iid)} cached items in {self.cache_dir}')
            return True
        return False

    def get_cache_path(self, iid: str):
        return self.cache_dir / f'{iid}.pt'

    def create_cache(self):
        import torch
        from tqdm import tqdm
        from torch.utils.data import DataLoader

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        dl = DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=list)
        for batch in tqdm(dl):
            for item in batch:
                item_id = item[self.id_key]
                item_path = self.get_cache_path(item_id)

                if not item_path.exists():
                    torch.save(item, item_path)

    def invalidate_cache(self):
        import shutil
        shutil.rmtree(self.cache_dir)
