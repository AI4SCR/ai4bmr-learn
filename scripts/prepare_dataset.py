from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
from ai4bmr_core.utils.saving import save_image, save_mask
from ai4bmr_core.utils.stats import StatsRecorder
from loguru import logger
from tqdm import tqdm

from ai4bmr_learn.data.splits import generate_splits
from ai4bmr_learn.utils import io


# %% HELPER
def normalize(img, censoring=0.999, cofactor=1, exclude_zeros=True):
    img = np.arcsinh(img / cofactor)

    if exclude_zeros:
        masked_img = np.where(img == 0, np.nan, img)
        thres = np.nanquantile(masked_img, censoring, axis=(1, 2), keepdims=True)
    else:
        thres = np.quantile(img, q=censoring, axis=(1, 2), keepdims=True)

    img = np.minimum(img, thres)

    return img


class PrepareDatasetFolder(L.LightningDataModule):

    def __init__(self,
                 dataset,
                 save_dir: Path,
                 coords_version: str = 'size=224-stride=224',
                 split_version: str | None = None,
                 split_kwargs: dict | None = None,
                 metadata_path: Path | None = None,
                 annotation_col_name: str | None = None,
                 force: bool = False,
                 ):
        super().__init__()

        self.force = force

        # DATASET
        self.dataset = dataset
        self.save_dir = save_dir.resolve() / dataset.name

        # IMAGES
        self.image_version = self.dataset.image_version
        self.images_dir = self.save_dir / 'images' / self.image_version

        # STATS
        self.stats_path = self.images_dir / 'stats.json'

        # MASKS
        self.mask_version = self.dataset.mask_version
        self.masks_dir = self.save_dir / 'masks' / self.mask_version

        # COORDS
        self.coords_version = coords_version
        self.coords_path = self.save_dir / 'coords' / f'{coords_version}.json'
        self.items_path = self.save_dir / 'items' / 'items.json'

        # METADATA
        self.metadata_path = metadata_path or self.save_dir / 'metadata.parquet'

        # ANNOTATIONS
        self.annotation_version = self.dataset.metadata_version
        self.annotation_col_name = annotation_col_name
        if annotation_col_name is not None:
            self.annotations_dir = self.save_dir / 'annotations' / annotation_col_name

        # SPLIT
        self.split_version = split_version
        if split_version is not None:
            self.split_kwargs = split_kwargs or {}
            self.splits_dir = self.save_dir / 'splits'

        self.save_hyperparameters()

    def prepare_splits(self):
        self.prepare_samples_splits()
        self.prepare_coords_split()

    def prepare_coords_split(self):
        if self.split_version is None:
            return

        splits_path = self.save_dir / 'splits' / 'coords' /f'{self.split_version}.parquet'
        if  splits_path.exists() and not self.force:
            return

        logger.info('Preparing coords splits...')

        splits_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = pd.read_parquet(self.metadata_path, engine='fastparquet')
        coords = pd.read_json(self.coords_path)
        coords['sample_id'] = coords.image_path.map(lambda x: Path(x).stem)

        intx = list(set(coords).intersection(set(metadata)))
        logger.warning(f'Metadata columns {intx} conflict with coord metadata. Dropping them.')
        coords = coords.merge(metadata.drop(columns=intx), how='left', left_on='sample_id', right_index=True)

        kwargs = dict(test_size=0.2, val_size=0.2, stratify=True, encode_targets=True, random_state=0)
        kwargs.update(self.split_kwargs)
        metadata = generate_splits(metadata=coords, **kwargs)
        metadata = metadata.set_index('uuid')
        metadata.to_parquet(splits_path, engine='fastparquet')
        metadata.groupby('image_path').size().rename('num_patches').to_csv(splits_path.with_suffix('.csv'))


    def prepare_samples_splits(self):

        if self.split_version is None:
            return

        splits_path = self.save_dir / 'splits' / 'samples' / f'{self.split_version}.parquet'
        if  splits_path.exists() and not self.force:
            return

        logger.info('Preparing samples splits...')

        splits_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = pd.read_parquet(self.metadata_path, engine='fastparquet')
        kwargs = dict(test_size=0.2, val_size=0.2, stratify=True, encode_targets=True, random_state=0)
        kwargs.update(self.split_kwargs)
        metadata = generate_splits(metadata=metadata, **kwargs)
        metadata.to_parquet(splits_path, engine='fastparquet')


    def prepare_annotation(self):
        if self.annotation_col_name is None or (self.annotations_dir.exists() and not self.force):
            return

        logger.info(f'Preparing annotations...')

        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        metadata = self.dataset.metadata
        image_ids = set([i.stem for i in self.images_dir.glob('*.zarr')])
        mask_ids = set([i.stem for i in self.masks_dir.glob('*.zarr')])
        metadata_ids = metadata.index.get_level_values('sample_id').unique()

        sample_ids = set(image_ids) & set(mask_ids) & set(metadata_ids)

        col_name = self.annotation_col_name
        label_to_id = {v: k for k, v in enumerate(metadata[col_name].unique(), start=1)}
        metadata['label_id'] = metadata[col_name].map(label_to_id)

        for sample_id in tqdm(sample_ids):
            obj_to_label = metadata.loc[sample_id, 'label_id']
            obj_to_label.index = obj_to_label.index.astype(int)
            obj_to_label = obj_to_label.to_dict()
            obj_to_label[0] = 0

            mask = io.imread(self.masks_dir / f'{sample_id}.zarr')
            assert set(mask.flatten()) == set(obj_to_label)
            labels = np.vectorize(obj_to_label.get)(mask)

            save_mask(labels, self.annotations_dir / f'{sample_id}.zarr')

    def prepare_metadata(self):
        if self.metadata_path.exists() and not self.force:
            return

        logger.info(f'Preparing metadata...')

        sample_ids = [i.stem for i in self.images_dir.glob('*.zarr')] + [i.stem for i in self.masks_dir.glob('*.zarr')]
        sample_ids = list(set(sample_ids))
        filter_ = self.dataset.clinical.index.isin(sample_ids)
        metadata = self.dataset.clinical[filter_]
        metadata = metadata.convert_dtypes()

        cols = metadata.select_dtypes(['datetime']).columns
        for col in cols:
            metadata[col] = metadata[col].dt.strftime("%Y-%m-%d")

        metadata.to_parquet(self.metadata_path, engine='fastparquet')

    def prepare_masks(self):
        if self.masks_dir.exists() and not self.force:
            return

        logger.info(f'Preparing masks...')

        self.masks_dir.mkdir(parents=True, exist_ok=True)
        sample_ids = sorted(set(self.dataset.masks))

        for sample_id in tqdm(sample_ids):
            mask = self.dataset.masks[sample_id].data
            mask = mask.astype(np.uint16) if mask.max() < 65534 else mask.astype(np.uint32)
            io.imsave(mask, self.masks_dir / f"{sample_id}.zarr")

    def prepare_images(self):
        if self.images_dir.exists() and self.stats_path.exists() and not self.force:
            return

        logger.info(f'Preparing images...')

        self.images_dir.mkdir(parents=True, exist_ok=True)
        sample_ids = sorted(set(self.dataset.images))

        # NORMALIZE IMAGES AND COMPUTE STATS
        sr = StatsRecorder()
        for sample_id in tqdm(sample_ids):
            image = self.dataset.images[sample_id].data
            image = normalize(image)
            sr.update(image)
            image = image.astype(np.float32)
            save_image(image, save_path=self.images_dir / f"{sample_id}.zarr", chunks=(image.shape[0], 512, 512))

        pd.Series(sr.__dict__).to_json(self.stats_path)

    def prepare_coords(self):
        from ai4bmr_learn.utils.images import get_coordinates_dict
        from ai4bmr_learn.data_models.Coordinate_v2 import PatchCoordinate
        import json

        if self.coords_path.exists() and not self.force:
            return

        logger.info(f'Preparing coords...')

        self.coords_path.parent.mkdir(parents=True, exist_ok=True)

        coords = []
        img_paths = sorted(self.images_dir.glob('*.zarr'))
        for img_path in tqdm(img_paths):
            img = io.imread(img_path)
            _, height, width = img.shape
            coords_dict = get_coordinates_dict(height=height, width=width, kernel_size=256, stride=256)
            # TODO: filter coords if necessary
            coords.extend(
                [PatchCoordinate(**i, image_path=str(img_path)).model_dump() for i in coords_dict]
            )

        with open(self.coords_path, 'w') as f:
            json.dump(coords, f)

        coords = pd.DataFrame.from_records(coords)
        coords.groupby('image_path').size().rename('num_patches').to_csv(self.coords_path.with_suffix('.csv'), index=True)

    def sanity_checks(self):
        image_ids = set([i.stem for i in self.images_dir.glob('*.zarr')])
        mask_ids = set([i.stem for i in self.masks_dir.glob('*.zarr')])
        if image_ids != mask_ids:
            logger.warning(f'Set of images and mask ids do not match')

    def prepare_items(self):
        import uuid
        import json

        if self.items_path.exists():
            return

        items = []
        for img_path in self.images_dir.glob('*.zarr'):
            items.append({'uuid': str(uuid.uuid4()), 'image_path': str(img_path), 'sample_id': img_path.stem})

        self.items_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.items_path, 'w') as f:
            json.dump(items, f)

    def prepare_data(self) -> None:

        self.dataset.setup()

        self.prepare_images()
        self.prepare_masks()
        self.prepare_coords()
        self.prepare_items()
        self.prepare_metadata()

        self.prepare_splits()
        self.prepare_annotation()

        self.sanity_checks()


# %%
# save_dir = Path('/users/amarti51/prometex/data/benchmarking/datasets')

# from ai4bmr_datasets import Cords2024
# splits_kwargs = dict(target_column_name='dx_name',
#                      include_targets=['Adenocarcinoma', 'Squamous cell carcinoma'],
#                      # encode_targets=False,
#                      use_filtered_targets_for_train=True)
# ds = Cords2024(image_version='published', mask_version='annotated')
# dm = self = PrepareDatasetFolder(dataset=ds, save_dir=save_dir, split_version='ssl-target=dx_name', split_kwargs=splits_kwargs)
# dm.prepare_data()
# dm.prepare_images()
# dm.prepare_masks()
# dm.prepare_coords()
# dm.prepare_splits()
#
# %%
# splits_kwargs = dict(target_column_name='dx_name',
#                      include_targets=['Adenocarcinoma', 'Squamous cell carcinoma'],
#                      use_filtered_targets_for_train=False)
# dm = PrepareDatasetFolder(dataset=ds, save_dir=save_dir,
#                           split_version='clf-target=dx_name', split_kwargs=splits_kwargs)
# dm.prepare_splits()
# dm.prepare_data()

# dm = self = PrepareDatasetFolder(dataset=Cords2024(), save_dir=save_dir,
#                           split_version='clf-target=dx_name', split_kwargs=splits_kwargs,
#                           annotation_version='annotated', annotation_col_name='cell_type')
# dm.prepare_data()

# from ai4bmr_datasets import PCa
# target_name = 'disease_progr'
# splits_kwargs = dict(target_column_name=target_name, use_filtered_targets_for_train=True)
# dm = PrepareDatasetFolder(dataset=PCa(), save_dir=save_dir,
#                           image_version='filtered', mask_version='annotated',
#                           split_version=f'ssl-target={target_name}', split_kwargs=splits_kwargs,
#                           annotation_version='filtered-annotated', annotation_col_name='label')
# dm.prepare_data()

# %% PREPARE ITEMS
# items = []
# import uuid
# for img_path in Path('/users/amarti51/prometex/data/benchmarking/datasets/Cords2024/images/published').glob('*.zarr'):
#     items.append({'uuid': str(uuid.uuid4()), 'image_path': str(img_path), 'sample_id': img_path.stem})
#
# import json
# save_path = Path('/users/amarti51/prometex/data/benchmarking/datasets/Cords2024/items/items.json')
# save_path.parent.mkdir(parents=True, exist_ok=True)
#
# with open(save_path, 'w') as f:
#     json.dump(items, f)

if __name__ == '__main__':
    from ai4bmr_datasets import PCa

    save_dir = Path('/users/amarti51/prometex/data/benchmarking/datasets')
    dataset = PCa(image_version='filtered', mask_version='annotated',
                  metadata_version='filtered-annotated', load_metadata=True)
    target_name = 'disease_progr'
    splits_kwargs = dict(target_column_name=target_name, use_filtered_targets_for_train=True)
    dm = self = PrepareDatasetFolder(dataset=dataset, save_dir=save_dir,
                              split_version=f'ssl-target={target_name}', split_kwargs=splits_kwargs,
                              annotation_col_name='label')
    # dm.prepare_data()

    splits_kwargs = dict(target_column_name=target_name, use_filtered_targets_for_train=False)
    dm = self = PrepareDatasetFolder(dataset=dataset, save_dir=save_dir,
                              split_version=f'clf-target={target_name}', split_kwargs=splits_kwargs,
                              annotation_col_name='label')
    # dm.prepare_data()
    dm.prepare_metadata()
    dm.prepare_splits()
