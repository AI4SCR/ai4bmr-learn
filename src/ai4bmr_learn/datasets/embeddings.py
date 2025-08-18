from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset

from ai4bmr_learn.datasets.utils import filter_items_and_metadata


class Embeddings(Dataset):

    def __init__(self, data_dir: Path | None = None, transform: Callable | None = None, pattern: str = '*.pt',
                 metadata_path: Path | None = None, id_key: str | None = None, drop_nan_columns: bool = False,
                 split: str | None = None):
        super().__init__()

        self.data_dir = data_dir.resolve()
        assert self.data_dir.exists(), f'{self.data_dir} does not exist.'
        self.pattern = pattern

        self.transform = transform
        self.data_paths: list[Path] | None = None

        # METADATA
        self.metadata_path = metadata_path
        self.id_key = id_key
        if metadata_path is not None:
            self.metadata_path = Path(metadata_path).expanduser().resolve()
            assert self.metadata_path.exists(), f'metadata_path {self.metadata_path} does not exist'
            assert self.id_key is not None, f'provide the `id_key` to look up metadata for an item.'
        self.metadata: pd.DataFrame | None = None
        self.drop_nan_columns = drop_nan_columns
        self.split = split

    def setup(self):
        self.data_paths = sorted(self.data_dir.glob(self.pattern))
        logger.info(f'Found {len(self.data_paths)} embeddings in {self.data_dir}')

        if self.metadata_path is not None:
            raise NotImplementedError('Metadata loading is not implemented yet.')
            items = []
            item_ids = [i[self.id_key] for i in items]
            metadata = pd.read_parquet(self.metadata_path)
            self.item_ids, self.metadata = filter_items_and_metadata(item_ids=item_ids, metadata=metadata,
                                                                     split=self.split,
                                                                     drop_nan_columns=self.drop_nan_columns)

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, idx: int):
        data_path = self.data_paths[idx]
        item = torch.load(str(data_path), map_location='cpu', weights_only=True)

        if self.metadata is not None:
            metadata_dict = self.metadata.loc[idx].to_dict()
            item['metadata'] = metadata_dict

        if self.transform:
            item = self.transform(item)

        return item

# ds = Embeddings(Path('/users/amarti51/prometex/data/benchmarking/embeddings/PCa/oee8hrej'))
# ds.setup()
