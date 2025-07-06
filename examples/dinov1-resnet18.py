# %%
from dotenv import load_dotenv

load_dotenv()

import torch
import lightning as L
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from ai4bmr_learn.datasets.imagenet import ImageNet

from ai4bmr_learn.transforms.dino_transform import DINOTransform

# %% CONFIGURATION
fast_dev_run = False
max_epochs = 300
num_workers = 16

# %% SSL
from ai4bmr_learn.ssl.dinov1 import DINOv1
import torchvision
import torch.nn as nn

resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])

input_dim = 512
batch_size = 64
backbone_name = 'resnet18'

ssl = DINOv1(backbone=backbone, input_dim=input_dim, max_epochs=max_epochs)

# DATA
from torch.utils.data import Subset
from torchvision.transforms import v2
from lightly.transforms.dino_transform import IMAGENET_NORMALIZE

train_transform = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    DINOTransform(),
    v2.Normalize(mean=IMAGENET_NORMALIZE['mean'], std=IMAGENET_NORMALIZE['std'])
])

val_transform = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((224, 224)),
    v2.Normalize(mean=IMAGENET_NORMALIZE['mean'], std=IMAGENET_NORMALIZE['std'])
])


ds = ImageNet()

num_classes = 10
max_index = 1300 * num_classes
targets = set(ds.dataset.targets[:max_index])
classes = [ds.idx_to_class[i] for i in targets]

g = torch.Generator().manual_seed(0)
indices = torch.randperm(max_index, generator=g)
max_train_index = int(max_index * 0.9)
train_indices = indices[:int(max_index * 0.9)]
val_indices = indices[int(max_index * 0.9):]

ds_train = Subset(ImageNet(transform=train_transform), indices=train_indices)
dl_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=num_workers,
)

ds_val = Subset(ImageNet(transform=val_transform), indices=val_indices)
dl_val = torch.utils.data.DataLoader(
    ds_val,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)

# LOGGER
from ai4bmr_learn.utils.utils import setup_wandb_auth
from ai4bmr_learn.utils.stats import model_stats

model_stats_dict = {f'student_backbone_{k}': v for k, v in model_stats(ssl.student_backbone).items()}
model_stats_dict.update({f'student_head_{k}': v for k, v in model_stats(ssl.student_head).items()})
model_stats_dict.update({f'teacher_backbone_{k}': v for k, v in model_stats(ssl.teacher_backbone).items()})
model_stats_dict.update({f'teacher_head_{k}': v for k, v in model_stats(ssl.teacher_head).items()})

# %% CONFIGURATIONS
from ai4bmr_learn.utils.names import generate_name

name = generate_name()

setup_wandb_auth()

config = {
    # 'dataset': 'vocdetection',
    'dataset': 'ImageNet',
    'backbone': backbone_name,
    'transform': 'DINOTransform',
}

# TODO: this is overwritten by wandb.init
save_dir = '/work/FAC/FBM/DBC/mrapsoma/prometex/data/mae/logs'
wandb_logger = WandbLogger(project='dinov1',
                           name=name,
                           config=config,
                           log_model=False,
                           save_dir=save_dir)

# CALLBACKS
from ai4bmr_learn.callbacks.ImageSamples import DINOImageSamples
from ai4bmr_learn.callbacks.UMAP import UMAP
from ai4bmr_learn.callbacks.LinearProbing import LinearProbing
lr_monitor = LearningRateMonitor(logging_interval="epoch")
image_samples = DINOImageSamples(num_samples=5)
umap = UMAP(label_key='target', log_before_train=True, log_every_num_epochs=10)
linear_probing = LinearProbing(target_key='target')

# TRAINER
trainer = L.Trainer(
    max_epochs=max_epochs,
    logger=wandb_logger,
    callbacks=[linear_probing, image_samples, umap, lr_monitor],
    fast_dev_run=fast_dev_run,
)

trainer.fit(model=ssl, train_dataloaders=dl_train, val_dataloaders=dl_val)
wandb.finish()

# %%
