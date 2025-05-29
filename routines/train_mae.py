# %% DATAMODULE
import lightning as L
from pathlib import Path
from loguru import logger
from ai4bmr_core.utils.saving import save_zarr
import pickle
from torch.utils.data import DataLoader


class Danenberg(L.LightningDataModule):

    def __init__(
            self,
            base_dir: Path = Path("/work/FAC/FBM/DBC/mrapsoma/prometex/data/mae/datasets/"),
            image_version: str = "v1_standard",
            batch_size: int = 64,
            num_workers: int = None,
            persistent_workers: bool = True,
            shuffle: bool = True,
            pin_memory: bool = True,
    ):
        from torch import get_num_threads

        super().__init__()

        # DATALOADERS
        self.batch_size = batch_size
        self.num_workers = (
            num_workers if num_workers is not None else get_num_threads() - 1
        )
        self.persistent_workers = persistent_workers if self.num_workers > 0 else False
        self.shuffle = shuffle
        self.pin_memory = pin_memory

        # DATASET
        self.base_dir = base_dir
        self.dataset_dir = base_dir / "Danenberg"
        self.image_version = image_version
        self.images_dir = self.dataset_dir / "images" / image_version

        self.coords_dir = self.dataset_dir / 'coords'
        self.splits_dir = self.dataset_dir / 'splits'
        self.metadata_path = self.dataset_dir / "metadata.parquet"

        self.train_idx, self.val_idx, self.test_idx = None, None, None
        self.train_set, self.val_set, self.test_set = None, None, None

    def setup(self, stage=None):
        from torch.utils.data import Subset
        pass
        # dataset = Patches(
        #     images_dir=self.images_dir,
        #     coords_path=self.random_coords_path,
        #     metadata_path=self.metadata_path,
        # )
        # splits = pd.read_parquet(self.splits_path)
        #
        # self.train_idx = np.flatnonzero(splits[Split.COLUMN_NAME] == Split.TRAIN)
        # self.val_idx = np.flatnonzero(splits[Split.COLUMN_NAME] == Split.VAL)
        # self.test_idx = np.flatnonzero(splits[Split.COLUMN_NAME] == Split.TEST)
        #
        # self.train_set, self.val_set, self.test_set = None, None, None
        #
        # self.train_set = Subset(dataset, self.train_idx)
        # self.val_set = Subset(dataset, self.val_idx)
        # self.test_set = Subset(dataset, self.test_idx)

    def preprocess_images(
            self,
            images,
            panel,
            save_dir: Path,
            include_channels: list[str] = None,
    ):
        from ai4bmr_core.utils.stats import StatsRecorder

        save_dir.mkdir(exist_ok=True, parents=True)
        save_stats = save_dir / "stats.pkl"
        save_panel = save_dir / "panel.parquet"

        if include_channels is not None:
            assert set(include_channels) <= set(panel.target), f"Invalid channels: {include_channels}"
            panel = panel[panel.target.isin(include_channels)]
            channel_idc = panel.index.values

            idx_name = panel.index.name
            panel = panel.reset_index(drop=True)
            panel.index.name = idx_name
        else:
            channel_idc = None

        panel.to_parquet(save_panel)

        sr = StatsRecorder()
        sample_ids = set()
        for i, img in enumerate(images.values(), start=1):
            logger.info(f"Processing {i}/{len(images)}")

            img_id = img.id
            img = img.data

            c, h, w = img.shape

            if h < 256 or w < 256:
                logger.warning(f"Ignoring small image: {img_id} with {h}x{w}")
                continue

            sample_ids.add(img_id)

            if channel_idc is not None:
                img = img[channel_idc]

            img = self.normalize(img)

            sr.update(img)  # track dataset stats

            save_path = save_dir / f"{img_id}.zarr"
            save_zarr(img, save_path=save_path)

        with open(save_stats, "wb") as f:
            pickle.dump(sr, f)
            logger.info(f'Saved dataset stats to: {save_stats}')

    def normalize(self, img):
        import numpy as np
        censoring = 0.99
        cofactor = 1
        exclude_zeros = True
        img = np.arcsinh(img / cofactor)
        if exclude_zeros:
            masked_img = np.where(img == 0, np.nan, img)
            thres = np.nanquantile(masked_img, censoring, axis=(1, 2), keepdims=True)
        else:
            thres = np.quantile(img, q=censoring, axis=(1, 2), keepdims=True)
        img = np.minimum(img, thres)
        return img

    def prepare_data(self) -> None:
        from pathlib import Path

        # DATASET
        from ai4bmr_datasets import Danenberg2022
        base_dir = Path("/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/Danenberg2022")
        dm = Danenberg2022(base_dir=base_dir)
        dm.setup(image_version='published', mask_version='published')

        # preprocessing
        if not self.images_dir.exists():
            self.preprocess_images(images=dm.images, panel=dm.panel, save_dir=self.images_dir)

        # metadata
        clinical = dm.clinical.copy()
        ids = {i.stem for i in self.images_dir.glob("*.zarr")}
        clinical = clinical.loc[list(ids)]
        clinical.to_parquet(self.metadata_path)

        # random training coords
        if not self.coords_dir.exists():
            self.prepare_coords()

        if not self.splits_dir.exists():
            self.prepare_splits()

    def prepare_coords(self):
        import zarr
        import numpy as np
        from collections import Counter
        from ai4bmr_learn.utils.images import get_random_coordinates_dict
        import json

        self.coords_dir.mkdir(exist_ok=True, parents=True)
        ids = {i.stem for i in self.images_dir.glob("*.zarr")}

        rng = np.random.default_rng(0)
        counter = Counter(rng.choice(list(ids), size=30_000, replace=True))
        for sample_id, count in counter.items():
            img_path = self.images_dir / f"{sample_id}.zarr"
            image = zarr.open_array(img_path, mode="r")
            c, h, w = image.shape

            coords_list = get_random_coordinates_dict(height=h, width=w, num_coords=count, kernel_size=224,
                                                      image_path=str(img_path), sample_id=sample_id)

            save_path = self.coords_dir / f"{sample_id}.json"
            with open(save_path, "w") as f:
                json.dump(coords_list, f)

    def prepare_splits(self):
        from ai4bmr_learn.data.splits import generate_splits, Split
        import json
        import pandas as pd

        coords = []
        for coords_file in self.coords_dir.glob("*.json"):
            with open(coords_file, "r") as f:
                sample_coords = json.load(f)
                coords.extend(sample_coords)
        metadata = pd.DataFrame.from_records(coords)
        splits = generate_splits(metadata, val_size=0.05, test_size=0.25, random_state=0)
        self.splits_dir.mkdir(exist_ok=True, parents=True)
        splits.to_parquet(self.splits_dir / "splits.parquet")
        for split, split_data in splits.groupby(Split.COLUMN_NAME, observed=True):
            split_coords = [coords[i] for i in split_data.index.values]
            with open(self.splits_dir / f"{split}.json", "w") as f:
                json.dump(split_coords, f)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )


dm = self = Danenberg()
dm.prepare_data()
