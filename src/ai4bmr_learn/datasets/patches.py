import json
from pathlib import Path
from typing import Callable
from torchvision import tv_tensors
from PIL.Image import Image

import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset

from ai4bmr_learn.data.splits import Split
from ai4bmr_learn.data_models.Coordinate_v2 import PatchCoordinate, SlideCoordinate
from ai4bmr_learn.utils import io
import numpy as np

from torchvision.transforms import v2
from ai4bmr_learn.utils.utils import pair


def get_patch(coord: PatchCoordinate) -> tv_tensors.Image:
    img_path = Path(coord.image_path)

    x, y = coord.x, coord.y
    kernel_height, kernel_width = pair(coord.kernel_size)
    level = coord.level if hasattr(coord, 'level') else 0

    patch = io.read_region(img_path=img_path, x=x, y=y, width=kernel_width, height=kernel_height, level=level)

    if isinstance(coord, SlideCoordinate):
        patch_height, patch_width = pair(coord.patch_size)
        scale_factor = coord.scale_factor

        assert np.isclose(kernel_width, patch_width * scale_factor)
        assert np.isclose(kernel_height, patch_height * scale_factor)
        assert np.isclose(coord.mpp * scale_factor, coord.patch_mpp)
        assert patch_height == round(kernel_height / scale_factor)
        assert patch_width == round(kernel_width / scale_factor)

        transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((patch_height, patch_width)),
        ])

    else:
        transform = v2.Compose([
            v2.ToImage(),
        ])

    patch = transform(patch)
    return patch


class Patches(Dataset):

    def __init__(
        self,
        coords_path: Path,
        metadata_path: Path | None = None,
        split: str | None = None,
        transform: Callable | None = None,
        cache_dir: Path | None = None,
        drop_nan_columns: bool = False,
    ):
        """
        Args:
            coords_path: Path to the JSON file containing the coordinates of the patches.
            metadata_path: Path to the Parquet file containing the metadata for the patches.
            transform: Transform to be applied to the patches and expression data.
        """
        super().__init__()

        # COORDS
        self.coords_path = Path(coords_path).expanduser().resolve()
        assert self.coords_path.exists(), f'coords_path {self.coords_path} does not exist'
        self.coords: list[PatchCoordinate] | None = None
        self.coord_ids: list[str] | None = None
        self.image_ids: list[str] | None = None

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


    def setup(self):
        logger.info(f'Setting up Patches dataset from coords_path: {self.coords_path}')

        with open(self.coords_path, 'r') as f:
            self.coords = [PatchCoordinate(**i) for i in json.load(f)]
            self.coord_ids = [i.uuid for i in self.coords]
            # FIX: this is not safe if coords for different location with the same name are loaded.
            self.image_ids = [Path(i.image_path).stem for i in self.coords]
            logger.info(f'Loaded {len(self.coords)} patches')

        if self.metadata_path is not None:
            self.metadata = pd.read_parquet(self.metadata_path)

            if self.split is not None:
                keep = self.metadata[Split.COLUMN_NAME.value] == self.split
                assert keep.sum() > 0, f'There are no coords that belong to `split={self.split}`'
                self.metadata = self.metadata[keep]

                valid_uuids = set(self.metadata.index)
                self.coords = list(filter(lambda x: x.uuid in valid_uuids, self.coords))
                self.coord_ids = [i.uuid for i in self.coords]
                assert len(self.metadata) == len(self.coords), f'Metadata and coords differ'
                logger.info(f'Filtered coords and metadata for `split={self.split}`. Found {len(self.coords)} patches.')

            filter_ = self.metadata.isna().any()
            cols_with_nan = self.metadata.columns[filter_].tolist()
            if filter_.any():
                self.metadata = self.metadata.drop(columns=cols_with_nan) if self.drop_nan_columns else self.metadata
                msg = f"Detected NaN values in the metadata which is incompatible with `torch.DataLoader`. Affected columns: {cols_with_nan}. "
                msg += "Dropping them." if self.drop_nan_columns else "Use `drop_nan_columns=True` to drop them."
                logger.warning(msg)

                if self.split is not None and Split.COLUMN_NAME.value in cols_with_nan:
                    logger.warning(f'Detected NaN in column `split`. This is most likely a bug.')

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
            uuids = set([coord.uuid for coord in self.coords])

            # TODO: report subset matches
            if uuids <= self.cached_ids:
                logger.info(f'Found all {len(uuids)} cached patches in {self.coords_path}')
                return True
            else:
                return False

    def get_cache_path(self, uuid: str):
        return self.cache_dir / f'{uuid}.pt'

    def create_cache(self):
        from tqdm import tqdm

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        num_coords = len(self.coords)
        for i in tqdm(range(num_coords), total=num_coords, desc='Caching patches'):
            coord = self.coords[i]
            patch_path = self.get_cache_path(coord.uuid)
            # points_path = self.cache_dir / f'{coord.uuid}.parquet'

            # if not patch_path.exists() or not points_path.exists():
            if not patch_path.exists():
                item = self[i]
                self.cache_dir = Path(item['image_path']).parent / 'patches'
                self.cache_dir.mkdir(parents=True, exist_ok=True)

                torch.save(item['images'], patch_path)
                # item['points'].to_parquet(points_path)

    def invalidate_cache(self):
        import shutil
        shutil.rmtree(self.cache_dir)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx) -> dict:
        coord = self.coords[idx]
        item = coord.model_dump()

        if self.has_cache(uuid=coord.uuid):
            patch_path = self.get_cache_path(coord.uuid)
            patch = torch.load(patch_path, weights_only=False)
        else:
            patch = get_patch(coord)

        item['images'] = patch

        if self.metadata is not None:
            metadata_dict = self.metadata.loc[coord.uuid].to_dict()
            item['metadata'] = metadata_dict

        if self.transform:
            item = self.transform(item)

        return item
