import json
from pathlib import Path
from typing import Callable

import geopandas as gpd
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset
from torchvision import tv_tensors

from ai4bmr_learn.data.splits import Split
from ai4bmr_learn.data_models.Coordinate import SlideCoordinate
from ai4bmr_learn.utils import io
from ai4bmr_learn.utils.utils import pair
from torchvision.transforms import v2

to_img = lambda x: tv_tensors.Image(x)

to_patch = lambda x, size: v2.Compose([to_img, v2.Resize(size=size)])(x)

def get_points(coord):
    from shapely.affinity import translate, scale

    kernel_height, kernel_width = pair(coord.kernel_size)

    xmin = coord.x
    ymin = coord.y
    xmax = xmin + kernel_width
    ymax = ymin + kernel_height
    bbox = (xmin, ymin, xmax, ymax)

    scale_factor = getattr(coord, 'scale_factor', 1)

    # check option: use_arrwo=True, not supported for gpkg
    points = gpd.read_file(coord.points_path, bbox=bbox)

    points['geometry'] = points['geometry'].map(
        lambda geom: scale(
            translate(geom=geom, xoff=-xmin, yoff=-ymin, zoff=0),
            xfact=1 / scale_factor,
            yfact=1 / scale_factor,
            origin=(0, 0)
        )
    )

    return points


def get_patch(coord: SlideCoordinate) -> tv_tensors.Image:
    # TODO: How should we handle Slide or Patch Coordinates?
    img_path = Path(coord.image_path)

    x, y = coord.x, coord.y
    kernel_height, kernel_width = pair(coord.kernel_size)
    level = coord.level if hasattr(coord, 'level') else 0
    patch_size = coord.patch_size

    patch = io.read_region(img_path=img_path, x=x, y=y, width=kernel_width, height=kernel_height, level=level)
    patch = to_patch(x=patch, size=patch_size)
    return patch


class Coordinates(Dataset):

    def __init__(
            self,
            coords_path: Path,
            metadata_path: Path | None = None,
            split: str | None = None,
            transform: Callable | None = None,
            cache_dir: Path | None = None,
            drop_nan_columns: bool = False,
            with_image: bool = True,
            with_points: bool = False,
            index_key: str | None = None
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
        self.coords: list[SlideCoordinate] | None = None
        self.coord_ids: list[str] | None = None
        self.image_ids: list[str] | None = None
        self.with_image = with_image
        self.with_points = with_points

        # METADATA
        self.metadata_path = metadata_path
        self.index_key = index_key
        if metadata_path is not None:
            self.metadata_path = Path(metadata_path).expanduser().resolve()
            assert self.metadata_path.exists(), f'metadata_path {self.metadata_path} does not exist'
            assert self.index_key is not None, f'provide the `index_key` to look up metadata for an item.'
        self.metadata: pd.DataFrame | None = None
        self.drop_nan_columns = drop_nan_columns
        self.split = split

        # CACHE
        self.cache_dir = cache_dir.resolve() if cache_dir else None
        self.cached_ids: list[str] | None = None

        # TRANSFORM
        self.transform = transform

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx) -> dict:
        coord = self.coords[idx]
        item = coord.model_dump()

        if self.has_cache(uuid=coord.uuid):

            if self.with_image:
                patch_path = self.cache_dir / f'{coord.uuid}.pt'
                patch = torch.load(patch_path, weights_only=False)
                item['image'] = patch

            if self.with_points:
                points_path = self.cache_dir / f'{coord.uuid}.parquet'
                points = gpd.read_parquet(points_path)
                item['points'] = points

        else:
            if self.with_image:
                patch = get_patch(coord)
                item['image'] = patch

            if self.with_points:
                points = get_points(coord)
                item['points'] = points

        if self.metadata is not None:
            index = item[self.index_key]
            metadata_dict = self.metadata.loc[index].to_dict()
            item['metadata'] = metadata_dict

        if self.transform:
            item = self.transform(item)

        return item

    def setup(self):
        logger.info(f'Setting up Patches dataset from coords_path: {self.coords_path}')

        with open(self.coords_path, 'r') as f:
            self.coords = [SlideCoordinate(**i) for i in json.load(f)]
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
            # TODO: this does not error if only the images or only the points are cached...
            self.cached_ids = set([i.stem for i in self.cache_dir.rglob('*.pt')])

        if uuid is not None:
            return uuid in self.cached_ids
        else:
            uuids = set([coord.uuid for coord in self.coords])

            # TODO: report subset matches
            if uuids <= self.cached_ids:
                logger.info(f'Found cache for all {len(uuids)} coordinates in {self.cache_dir}')
                return True
            else:
                return False

    def get_cache_path(self, uuid: str):
        return self.cache_dir / f'{uuid}.pt'

    def create_cache(self):
        from tqdm import tqdm

        if self.transform is not None:
            logger.warning(f'caching items while using a transform.')

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        num_coords = len(self.coords)
        for i in tqdm(range(num_coords), total=num_coords, desc='Caching patches'):
            coord = self.coords[i]
            patch_path = self.cache_dir / f'{coord.uuid}.pt'
            points_path = self.cache_dir / f'{coord.uuid}.parquet'

            # if not patch_path.exists() or not points_path.exists():
            if not patch_path.exists():
                item = self[i]
                torch.save(item['image'], patch_path)
                item['points'].to_parquet(points_path)

    def invalidate_cache(self):
        import shutil
        shutil.rmtree(self.cache_dir)

    @classmethod
    def from_list(cls, coords: list[SlideCoordinate], **kwargs):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            coords_path = Path(tmpdir) / 'coords.json'
            with open(coords_path, 'w') as f:
                json.dump([coord.model_dump() for coord in coords], f)

            instance = cls(coords_path=coords_path, **kwargs)
            instance.setup()
            return instance
