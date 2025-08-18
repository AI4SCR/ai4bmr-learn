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

to_img = lambda x: tv_tensors.Image(x)


def get_patch(item: dict) -> tv_tensors.Image:
    img_path = Path(item['image_path'])

    x, y = item['x'], item['y']
    kernel_height, kernel_width = pair(item['kernel_size'])
    level = item['level'] if 'level' in item else 0

    patch = io.read_region(img_path=img_path, x=x, y=y, width=kernel_width, height=kernel_height, level=level)
    return to_img(patch)


def get_image(item: dict) -> tv_tensors.Image:
    img_path = Path(item['image_path'])
    patch = io.imread(img_path=img_path)
    return to_img(patch)


def get_mask(item):
    pass


def get_graph(item):
    pass


def get_annotation(item):
    pass


class Items(Dataset):

    def __init__(
            self,
            items_path: Path,
            metadata_path: Path | None = None,
            split: str | None = None,
            transform: Callable | None = None,
            cache_dir: Path | None = None,
            drop_nan_columns: bool = False,
            id_key: str = 'uuid'
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

        # METADATA
        self.metadata_path = metadata_path
        if metadata_path is not None:
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
        item = deepcopy(self.items[idx])
        # item = item.model_dump()

        if self.has_cache(uuid=item[self.id_key]):
            cache_path = self.get_cache_path(item[self.id_key])
            item = torch.load(cache_path, weights_only=False)
            return item
        else:
            image = get_image(item)
            item['image'] = image

        if self.metadata is not None:
            metadata_dict = self.metadata.loc[idx].to_dict()
            item['metadata'] = metadata_dict

        if self.transform:
            item = self.transform(item)

        return item

    def setup(self):
        logger.info(f'Setting up Items dataset from items_path: {self.items_path}')

        with open(self.items_path, 'r') as f:
            # self.items = [PatchCoordinate(**i) for i in json.load(f)]
            self.items = json.load(f)
            self.item_ids = [i[self.id_key] for i in self.items]
            logger.info(f'Loaded {len(self.items)} items')

        if self.metadata_path is not None:
            item_ids = [i[self.id_key] for i in self.items]
            metadata = pd.read_parquet(self.metadata_path)
            self.item_ids, self.metadata = filter_items_and_metadata(item_ids=item_ids, metadata=metadata,
                                                                     split=self.split,
                                                                     drop_nan_columns=self.drop_nan_columns)
            self.items = [i for i in self.items if i[self.id_key] in self.item_ids]

        if self.cache_dir and not self.has_cache():
            logger.info('No cache found. Creating...')
            self.create_cache()

    def has_cache(self, uuid: str | None = None) -> bool:

        if self.cache_dir is None:
            return False

        if self.cached_ids is None:
            logger.info(f'Gather cached ids...')
            self.cached_ids = set([i.stem for i in self.cache_dir.rglob('*.pt')])

        if uuid is not None:
            return uuid in self.cached_ids
        else:
            uuids = set([item[self.id_key] for item in self.items])

            # TODO: report subset matches
            if uuids <= self.cached_ids:
                logger.info(f'Found all {len(uuids)} cached patches in {self.items_path}')
                return True
            else:
                return False

    def get_cache_path(self, uuid: str):
        return self.cache_dir / f'{uuid}.pt'

    def create_cache(self):
        from tqdm import tqdm

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        num_items = len(self.items)
        for i in tqdm(range(num_items), total=num_items, desc='Caching items'):
            item = self.items[i]
            item_path = self.get_cache_path(item[self.id_key])

            if not item_path.exists():
                item = self[i]
                torch.save(item['image'], item_path)

    def invalidate_cache(self):
        import shutil
        shutil.rmtree(self.cache_dir)

# items = self = Items(
#     items_path=Path('/users/amarti51/prometex/data/benchmarking/datasets/Cords2024/items/items.json'),
#     metadata_path=Path('/users/amarti51/prometex/data/benchmarking/datasets/Cords2024/splits/samples/clf-target=dx_name.parquet'),
#     id_key='sample_id',
#     split='fit',
#     drop_nan_columns=True
# )
# items.setup()
# items[0]['image']
