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


class Items(Dataset):
    name: str = 'Items'

    def __init__(
            self,
            items_path: Path,
            metadata_path: Path | None = None,
            metadata_keys: list[str] | None = None,
            split: str | None = None,
            transform: Callable | None = None,
            cache_dir: Path | None = None,
            drop_nan_columns: bool = False,
            id_key: str | None = None,
    ):
        """
        Args:
            items_path: Path to the JSON file containing the coordinates of the patches.
            metadata_path: Path to the Parquet file containing the metadata for the patches.
            transform: Transform to be applied to the patches and expression data.
        """
        super().__init__()

        # ITEMS
        self.items_path = Path(items_path).expanduser().resolve()
        assert self.items_path.exists(), f'items_path {self.items_path} does not exist'
        self.items: list[dict] | None = None
        self.id_key = id_key
        self.item_ids: list[str | int] | None = None

        # METADATA
        self.metadata_path = metadata_path
        self.metadata_keys = metadata_keys
        if metadata_path is not None:
            assert id_key is not None, 'id_key must be provided when metadata_path is provided'
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
        return len(self.items)

    def __getitem__(self, idx) -> dict:
        raise NotImplementedError('Use either `Images`, `Patches` or inherit to create your own subclass.')

    def setup(self):
        logger.info(f'Setting up {self.name} dataset from items_path: {self.items_path}')

        with open(self.items_path, 'r') as f:
            # self.items = [PatchCoordinate(**i) for i in json.load(f)]
            self.items = json.load(f)
            self.item_ids = [i[self.id_key] for i in self.items] if self.id_key else None
            logger.info(f'Loaded {len(self.items)} items')

        if self.metadata_path is not None:
            logger.info(f'Loading metadata from {self.metadata_path}')
            item_ids = [i[self.id_key] for i in self.items]
            metadata = pd.read_parquet(self.metadata_path)

            if self.metadata_keys is not None:
                missing_keys = set(self.metadata_keys) - set(metadata.columns)
                assert len(missing_keys) == 0, f'Metadata keys {missing_keys} not found in metadata columns {metadata.columns}'
                metadata = metadata[self.metadata_keys]

            self.item_ids, self.metadata = filter_items_and_metadata(item_ids=item_ids, metadata=metadata,
                                                                     split=self.split,
                                                                     drop_nan_columns=self.drop_nan_columns)
            n = len(self.items)
            self.items = [i for i in self.items if i[self.id_key] in self.item_ids]
            logger.info(f'Loaded {len(self.items)}/{n} items have metadata')

        if self.cache_dir and not self.has_cache():
            logger.info(f'No cache found. Creating at {self.cache_dir}')
            self.create_cache()

    def has_cache(self, iid: str | None = None) -> bool:

        if self.cache_dir is None:
            return False

        if self.cached_ids is None:
            logger.info(f'Gather cached ids...')
            self.cached_ids = set([i.stem for i in self.cache_dir.rglob('*.pt')])

        if iid is not None:
            return iid in self.cached_ids
        else:
            iid = set([item[self.id_key] for item in self.items])

            # TODO: report subset matches
            if iid <= self.cached_ids:
                logger.info(f'Found all {len(iid)} cached items in {self.cache_dir}')
                return True
            else:
                return False

    def get_cache_path(self, iid: str):
        return self.cache_dir / f'{iid}.pt'

    def create_cache(self):
        from tqdm import tqdm

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        num_items = len(self.items)
        for i in tqdm(range(num_items), total=num_items, desc=f'Caching {self.name}'):
            item = self.items[i]
            item_path = self.get_cache_path(item[self.id_key])

            if not item_path.exists():
                item = self[i]
                torch.save(item, item_path)

    def invalidate_cache(self):
        import shutil
        shutil.rmtree(self.cache_dir)


def get_patch(item: dict, to_tensor: bool = True, to_patch_size: bool = False) -> Image | tv_tensors.Image:
    img_path = Path(item['image_path'])

    x, y = item['x'], item['y']
    kernel_height, kernel_width = pair(item['kernel_size'])
    level = item['level'] if 'level' in item else 0

    patch = io.read_region(img_path=img_path, x=x, y=y, width=kernel_width, height=kernel_height, level=level)

    transform = []
    if to_tensor:
        transform.append(v2.ToImage())

    if to_patch_size:
        patch_size = item['patch_size']
        transform.append(v2.Resize(size=patch_size))

    if len(transform) > 0:
        transform = v2.Compose(transform)
        return transform(patch)
    else:
        return patch


def get_image(item: dict) -> tv_tensors.Image:
    img_path = Path(item['image_path'])
    patch = io.imread(img_path=img_path)
    return tv_tensors.Image(patch)


def get_mask(item):
    pass


def get_graph(item):
    pass


def get_annotation(item):
    pass


class Images(Items):
    name: str = 'Images'

    def __getitem__(self, idx) -> dict:
        item = deepcopy(self.items[idx])
        iid = item[self.id_key]
        # item = item.model_dump()

        if self.has_cache(iid=iid):
            cache_path = self.get_cache_path(iid)
            item = torch.load(cache_path, weights_only=False)
            return item
        else:
            image = get_image(item)
            item['image'] = image

        if self.metadata is not None:
            metadata_dict = self.metadata.loc[iid].to_dict()
            item['metadata'] = metadata_dict

        if self.transform:
            item = self.transform(item)

        assert item['image'].shape[0] < 45
        return item


class Patches(Items):
    name: str = 'Patches'

    def __getitem__(self, idx) -> dict:
        item = deepcopy(self.items[idx])
        iid = item[self.id_key] if self.id_key else None
        # item = item.model_dump()

        if self.cache_dir is not None and self.has_cache(iid=iid):
            cache_path = self.get_cache_path(iid)
            item = torch.load(cache_path, weights_only=False)
            return item
        else:
            image = get_patch(item)
            item['image'] = image

        if self.metadata is not None:
            metadata_dict = self.metadata.loc[iid].to_dict()
            item['metadata'] = metadata_dict

        if self.transform:
            item = self.transform(item)

        return item


class SlidePatches(Items):
    name: str = 'SlidePatches'

    def __getitem__(self, idx) -> dict:
        item = deepcopy(self.items[idx])
        iid = item[self.id_key] if self.id_key else None

        if self.cache_dir is not None and self.has_cache(iid=iid):
            cache_path = self.get_cache_path(iid)
            item = torch.load(cache_path, weights_only=False)
            return item
        else:
            image = get_patch(item, to_tensor=True, to_patch_size=True)
            item['image'] = image

        if self.metadata is not None:
            metadata_dict = self.metadata.loc[iid].to_dict()
            item['metadata'] = metadata_dict

        if self.transform:
            item = self.transform(item)

        return item
