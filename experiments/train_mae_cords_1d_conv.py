# %%
import torch
import json
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.tv_tensors import Image
from dataclasses import asdict
from ai4bmr_learn.data_models.Coordinate import RandomCoordinate
from ai4bmr_learn.data_models.lightning_training import ProjectConfig, TrainerConfig, TrainingConfig, WandbInitConfig
import lightning as L
from ai4bmr_learn.models.backbones.base_backbone import BaseBackbone
from ai4bmr_learn.models.backbones.multi_to_rgb_vit import MultiChannelVit
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning import seed_everything
import lightning as L
from pathlib import Path
from loguru import logger
from ai4bmr_core.utils.saving import save_zarr
import pickle
from torch.utils.data import DataLoader
from dotenv import load_dotenv

load_dotenv()


# %% HELPER
def normalize(img, censoring=0.99, cofactor=1, exclude_zeros=True):
    img = np.arcsinh(img / cofactor)

    if exclude_zeros:
        masked_img = np.where(img == 0, np.nan, img)
        thres = np.nanquantile(masked_img, censoring, axis=(1, 2), keepdims=True)
    else:
        thres = np.quantile(img, q=censoring, axis=(1, 2), keepdims=True)

    img = np.minimum(img, thres)

    return img


def preprocess_images(
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

    # %%
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

        img = normalize(img)

        sr.update(img)  # track dataset stats

        save_path = save_dir / f"{img_id}.zarr"
        save_zarr(img, save_path=save_path)

    with open(save_stats, "wb") as f:
        pickle.dump(sr, f)
        logger.info(f'Saved dataset stats to: {save_stats}')


# %% DATASET
def get_patch(coord):
    import zarr
    from ai4bmr_learn.utils.utils import pair

    img_path = coord.image_path
    img = zarr.open_array(img_path)

    x, y = coord.x, coord.y
    kh, kw = pair(coord.kernel_size)

    patch = img[:, y: y + kh, x: x + kw]
    return patch


class Patches(Dataset):

    def __init__(
            self,
            images_dir: Path,
            coords_path: Path,
            metadata_path: Path,
            normalize: str = "dataset-level",
    ):

        self.coords_path = coords_path
        self.metadata_path = metadata_path
        self.panel_path = images_dir / "panel.parquet"

        self.normalize = normalize

        with open(self.coords_path, "r") as f:
            self.coords = json.load(f)

        self.metadata = pd.read_parquet(metadata_path)
        self.panel = pd.read_parquet(self.panel_path)

        self.num_channels = len(self.panel)
        self.image_size = self.coords[0]['kernel_size']  # assuming all patches have the same size

        self.images_dir = images_dir
        self.stats_path = self.images_dir / "stats.pkl"

        transform_norm = self.get_normalize_transform()
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=False),
                v2.GaussianBlur(kernel_size=(3, 3), sigma=1.0),
                transform_norm,
            ]
        )

    def get_normalize_transform(self):
        if self.normalize == "dataset-level":
            assert (
                self.stats_path.exists()
            ), f"No dataset level stats found at: {self.stats_path}"
            with open(self.stats_path, "rb") as f:
                sr = pickle.load(f)
                return v2.Normalize(mean=sr.mean, std=sr.std)
        elif self.normalize == "sample-level":
            raise NotImplementedError()
        else:
            return v2.Identity()

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        coord = RandomCoordinate(**coord)

        patch = get_patch(coord)
        patch = Image(patch)
        assert len(patch) == len(self.panel)
        assert patch.shape[1:] == (self.image_size, self.image_size)

        channel_names = self.panel.target.to_list()
        data = {'image': patch, 'channel_names': channel_names, **asdict(coord)}

        # sample_id = coord.sample_id
        # metadata = self.metadata.loc[sample_id].dropna().to_dict()
        # data = {'image': patch, 'metadata': metadata, 'channel_names': channel_names, **asdict(coord)}

        data = self.transform(data)

        return data


# %% DATAMODULE
class Cords2024(L.LightningDataModule):

    def __init__(
            self,
            base_dir: Path = Path("/work/FAC/FBM/DBC/mrapsoma/prometex/data/mae/datasets/"),
            image_version: str = "v1_standard",
            coords_version: str = 'num_coords=30000',
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
        self.dataset_dir = base_dir / "Cords2024"
        self.image_version = image_version
        self.images_dir = self.dataset_dir / "images" / image_version

        self.coords_version = coords_version
        self.coords_dir = self.dataset_dir / 'coords' / coords_version
        self.splits_dir = self.dataset_dir / 'splits' / coords_version
        self.metadata_path = self.dataset_dir / "metadata.parquet"

        self.train_set, self.val_set, self.test_set = None, None, None

    def setup(self, stage=None):

        self.train_set = Patches(
            images_dir=self.images_dir,
            coords_path=self.splits_dir / 'train.json',
            metadata_path=self.metadata_path,
        )

        self.test_set = Patches(
            images_dir=self.images_dir,
            coords_path=self.splits_dir / 'test.json',
            metadata_path=self.metadata_path,
        )

        self.val_set = Patches(
            images_dir=self.images_dir,
            coords_path=self.splits_dir / 'val.json',
            metadata_path=self.metadata_path,
        )

    def prepare_data(self) -> None:
        from pathlib import Path

        # DATASET
        from ai4bmr_datasets import Cords2024
        base_dir = Path("/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/Cords2024")
        dm = Cords2024(base_dir=base_dir)
        dm.setup(image_version='published', mask_version='published')

        # preprocessing
        if not self.images_dir.exists():
            preprocess_images(images=dm.images, panel=dm.panel, save_dir=self.images_dir)

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

            coords_dict_list = get_random_coordinates_dict(height=h, width=w, num_coords=count, kernel_size=224,
                                                           image_path=str(img_path), sample_id=sample_id)
            coords_list = [RandomCoordinate(**i) for i in coords_dict_list]
            coords_list = [asdict(i) for i in coords_list]

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


dm = self = Cords2024(num_workers=12)
dm.prepare_data()
dm.setup()

# %% BACKBONE
image_size = dm.train_set.image_size
num_channels = dm.train_set.num_channels

model_name = 'vit_small_patch16_224'
backbone = MultiChannelVit.from_timm_vit(model_name=model_name,
                                         image_size=image_size,
                                         num_channels=num_channels,
                                         pretrained=True,
                                         freeze_encoder=True)

# inp = torch.randn(1, 43, 224, 224)
# out = backbone(inp)

# %% CONFIGURATIONS
project_cfg = ProjectConfig(name="mae-cords2024")
trainer_cfg = TrainerConfig(max_epochs=1000,
                            accumulate_grad_batches=32,
                            # precision=16,
                            gradient_clip_val=1,
                            fast_dev_run=False)
training_cfg = TrainingConfig()
wandb_cfg = WandbInitConfig(project=project_cfg.name)

# %% SSL
from ai4bmr_learn.ssl.mae import MAE

decoder_kwargs = {'num_layers': 8, 'num_heads': 8, 'dim': 192}
ssl = MAE(backbone=backbone, decoder_kwargs=decoder_kwargs,
          batch_size=dm.batch_size, accumulate_grad_batches=trainer_cfg.accumulate_grad_batches)

# LOGGER
from ai4bmr_learn.utils.utils import setup_wandb_auth
from ai4bmr_learn.utils.stats import model_stats

# TODO: split in tokenizer, encoder, proj, decoder, head
model_stats_dict = {f'backbone_{k}': v for k, v in model_stats(ssl.backbone).items()}
model_stats_dict.update({f'decoder_{k}': v for k, v in model_stats(ssl.decoder).items()})

setup_wandb_auth()

metadata = {}
if not trainer_cfg.fast_dev_run:
    import wandb

    wandb.init(**asdict(wandb_cfg))
    ckpt_dir = project_cfg.ckpt_dir / wandb.run.name
    metadata["ckpt_dir"] = ckpt_dir
    wandb.config.update(metadata)
else:
    ckpt_dir = None

wandb_logger = WandbLogger(project=project_cfg.name, log_model=False, save_dir=project_cfg.log_dir)

# CALLBACKS
monitor_metric_name = "val_loss_epoch"
filename = "{epoch:02d}-{val_loss:.4f}"
model_ckpt = ModelCheckpoint(
    dirpath=ckpt_dir,
    monitor=monitor_metric_name,
    mode="min",
    save_top_k=1,
    filename=filename,
)
lr_monitor = LearningRateMonitor(logging_interval="epoch")
# early_stop = EarlyStopping(monitor=monitor_metric_name, mode='min', patience=50)

# TRAINER
torch.set_float32_matmul_precision('medium')
trainer = L.Trainer(
    logger=wandb_logger,
    # callbacks=[model_ckpt, lr_monitor, early_stop, run_info],
    callbacks=[model_ckpt, lr_monitor],
    **asdict(trainer_cfg),
)

# TRAIN
ckpt_path = training_cfg.ckpt_path  # resume from checkpoint
seed_everything(42, workers=True)
trainer.fit(model=ssl, datamodule=dm, ckpt_path=ckpt_path)

# Finish the wandb run
wandb.finish()
