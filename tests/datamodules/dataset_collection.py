from ai4bmr_learn.datamodules.dataset_collection import DatasetCollection
from ai4bmr_learn.datasets.dataset_folder import DatasetFolder
from pathlib import Path
from torchvision.transforms import v2
transform = v2.Resize((224, 224))

ds_fit = DatasetFolder(
    dataset_dir=Path('/users/amarti51/prometex/data/dinov1/datasets/Cords2024'),
    image_version='default',
    transform=transform,
    split_version='clf', split='fit')
ds_val = DatasetFolder(dataset_dir=Path('/users/amarti51/prometex/data/dinov1/datasets/Cords2024'),
                       target_name='dx_name', split_version='clf', split='val')

dc = DatasetCollection(datasets=dict(fit=ds_fit, val=ds_val), batch_size=4)
dc.setup()
train_batch = next(iter(dc.train_dataloader()))
assert 'sample_id' in train_batch
assert 'image' in train_batch
assert train_batch['image'].shape == (4, 43, 224, 224)

val_batch = next(iter(dc.val_dataloader()))
assert 'sample_id' in val_batch
assert 'image' not in val_batch

# %%
from pathlib import Path
import torch
from ai4bmr_learn.datasets.dataset_folder import DatasetFolder
from ai4bmr_learn.datamodules.dataset_collection import DatasetCollection
from torchvision.transforms.v2 import Compose, ToDtype, RandomResizedCrop
from ai4bmr_learn.transforms.dino_transform import DINOTransform
from ai4bmr_learn.transforms.wrappers import Normalize

# Constants
IMAGE_SIZE  = 224
BATCH_SIZE  = 8
NUM_WORKERS = 16
DATA_ROOT   = Path("/users/amarti51/prometex/data/dinov1/datasets/Cords2024")
STATS_PATH  = f"{DATA_ROOT}/images/default/stats.json"

# — “fit” split dataset —
fit_ds = DatasetFolder(
    dataset_dir=DATA_ROOT,
    image_version="default",
    split_version="ssl",
    split="fit",
    transform=Compose([
        ToDtype(dtype=torch.float32, scale=True),
        DINOTransform(global_crop_size=IMAGE_SIZE),
        Normalize(stats_path=STATS_PATH),
    ])
)

# — “val” split dataset —
val_ds = DatasetFolder(
    dataset_dir=DATA_ROOT,
    image_version="default",
    split_version="ssl",
    split="val",
    target_name="dx_name",
    transform=Compose([
        ToDtype(dtype=torch.float32, scale=True),
        RandomResizedCrop(size=IMAGE_SIZE, scale=(0.8, 1.0)),
        Normalize(stats_path=STATS_PATH),
    ])
)

# — DatasetCollection instance —
datamodule = DatasetCollection(
    datasets={
        "fit": fit_ds,
        "val": val_ds
    },
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    shuffle=True,
    pin_memory=True
)

dl = datamodule.train_dataloader()
datamodule.fit_set[0]
next(iter(dl))

item = datamodule.val_set[1]
item.keys()
dl = datamodule.val_dataloader()
for i, batch in enumerate(dl):
    print(i)
