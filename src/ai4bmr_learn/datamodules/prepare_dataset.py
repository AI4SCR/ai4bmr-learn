from pathlib import Path

import igraph
from tqdm import tqdm

import lightning as L
import numpy as np
import pandas as pd
import tifffile
from ai4bmr_core.utils.saving import save_image, save_mask
from ai4bmr_core.utils.stats import StatsRecorder
from loguru import logger
from ai4bmr_learn.data.splits import generate_splits
from tqdm import tqdm


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
                 image_version: str = 'published',
                 mask_version: str = 'published',
                 split_version: str | None = None,
                 split_kwargs: dict | None = None,
                 metadata_path: Path | None = None,
                 annotation_version: str | None = None,
                 annotation_col_name: str | None = None,
                 force: bool = False,
                 ):
        super().__init__()

        self.force = force

        # DATASET
        self.dataset = dataset
        self.save_dir = save_dir.resolve() / dataset.name

        # IMAGES
        self.image_version = image_version
        self.images_dir = self.save_dir / 'images' / self.image_version

        # STATS
        self.stats_path = self.images_dir / 'stats.json'

        # MASKS
        self.mask_version = mask_version
        self.masks_dir = self.save_dir / 'masks' / mask_version

        # METADATA
        self.metadata_path = metadata_path or self.save_dir / 'metadata.parquet'

        # ANNOTATIONS
        self.annotation_version = annotation_version
        self.annotation_col_name = annotation_col_name
        if annotation_col_name is not None:
            self.annotations_dir = self.save_dir / 'annotations' / annotation_col_name

        # SPLIT
        self.split_version = split_version
        if split_version is not None:
            self.split_kwargs = split_kwargs or {}
            self.splits_dir = self.save_dir / 'splits'
            self.splits_path = self.save_dir / 'splits' / f'{self.split_version}.parquet'

        self.save_hyperparameters()

    def prepare_splits(self):

        if self.split_version is None or (self.splits_path.exists() and not self.force):
            return

        logger.info('Preparing splits...')

        self.splits_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = pd.read_parquet(self.metadata_path, engine='fastparquet')
        kwargs = dict(test_size=0.2, val_size=0.2, stratify=True, encode_targets=True, random_state=0)
        kwargs.update(self.split_kwargs)
        metadata = generate_splits(metadata=metadata, **kwargs)
        metadata.to_parquet(self.splits_path, engine='fastparquet')

    def prepare_annotation(self):
        if self.annotation_col_name is None or (self.annotations_dir.exists() and not self.force):
            return

        logger.info(f'Preparing annotations...')

        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        metadata = self.dataset.metadata
        image_ids = set([i.stem for i in self.images_dir.glob('*.tiff')])
        mask_ids = set([i.stem for i in self.masks_dir.glob('*.tiff')])
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

            mask = tifffile.imread(self.masks_dir / f'{sample_id}.tiff')
            assert set(mask.flatten()) == set(obj_to_label)
            labels = np.vectorize(obj_to_label.get)(mask)

            save_mask(labels, self.annotations_dir / f'{sample_id}.tiff')

    def prepare_metadata(self):
        if self.metadata_path.exists() and not self.force:
            return

        logger.info(f'Preparing metadata...')
        sample_ids = [i.stem for i in self.images_dir.glob('*.tiff')] + [i.stem for i in self.masks_dir.glob('*.tiff')]
        sample_ids = list(set(sample_ids))
        filter_ = self.dataset.clinical.index.isin(sample_ids)
        self.dataset.clinical[filter_].to_parquet(self.metadata_path, engine='fastparquet')

    def prepare_masks(self):
        if self.masks_dir.exists() and not self.force:
            return

        logger.info(f'Preparing masks...')

        self.masks_dir.mkdir(parents=True, exist_ok=True)
        sample_ids = sorted(set(self.dataset.masks))

        for sample_id in tqdm(sample_ids):
            mask = self.dataset.masks[sample_id].data
            save_mask(mask, self.masks_dir / f"{sample_id}.tiff")

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
            save_image(image, self.images_dir / f"{sample_id}.tiff")

        pd.Series(sr.__dict__).to_json(self.stats_path)

    def sanity_checks(self):
        image_ids = set([i.stem for i in self.images_dir.glob('*.tiff')])
        mask_ids = set([i.stem for i in self.masks_dir.glob('*.tiff')])
        if image_ids != mask_ids:
            logger.warning(f'Set of images and mask ids do not match')

    def prepare_data(self) -> None:

        load_metadata = self.annotation_col_name is not None
        self.dataset.setup(image_version=self.image_version, mask_version=self.mask_version,
                           metadata_version=self.annotation_version, load_metadata=load_metadata)

        self.prepare_images()
        self.prepare_masks()
        self.prepare_metadata()

        self.prepare_splits()
        self.prepare_annotation()

        self.sanity_checks()


from ai4bmr_datasets import Cords2024
# generate_splits()
splits_kwargs = dict(target_column_name='dx_name',
                     include_targets=['Adenocarcinoma', 'Squamous cell carcinoma'],
                     use_filtered_targets_for_train=True)
save_dir = Path('/users/amarti51/prometex/data/benchmarking/datasets')
dm = PrepareDatasetFolder(dataset=Cords2024(), save_dir=save_dir,
                          split_version='ssl-target=dx_name', split_kwargs=splits_kwargs)
dm.prepare_data()

splits_kwargs = dict(target_column_name='dx_name',
                     include_targets=['Adenocarcinoma', 'Squamous cell carcinoma'],
                     use_filtered_targets_for_train=False)
save_dir = Path('/users/amarti51/prometex/data/benchmarking/datasets')
dm = PrepareDatasetFolder(dataset=Cords2024(), save_dir=save_dir,
                          split_version='clf-target=dx_name', split_kwargs=splits_kwargs)
dm.prepare_data()

dm = self = PrepareDatasetFolder(dataset=Cords2024(), save_dir=save_dir,
                          split_version='clf-target=dx_name', split_kwargs=splits_kwargs,
                          annotation_version='annotated', annotation_col_name='cell_type')
dm.prepare_data()
