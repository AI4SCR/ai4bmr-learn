# %%
from dataclasses import asdict

from dotenv import load_dotenv

load_dotenv()

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from ai4bmr_learn.datasets.vocdetection import VOCDetection

from ai4bmr_learn.transforms.dino_transform import DINOTransformLightly

# %% SSL
from ai4bmr_learn.ssl.dinov1 import DINOv1
import torchvision
import torch
import torch.nn as nn
import torchinfo
fast_dev_run = False
max_epochs = 300
# input_dim = 512
# batch_size = 64
# resnet = torchvision.models.resnet18()
# backbone = nn.Sequential(*list(resnet.children())[:-1])
# backbone_name = 'resnet18'

# instead of a resnet you can also use a vision transformer backbone as in the
# original paper (you might have to reduce the batch size in this case):
# backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
# torchinfo.summary(backbone)
#
# inp = torch.randn((1, 3, 224, 224))
# out = backbone(inp)
# out.shape
#
# input_dim = backbone.embed_dim
# batch_size = 32
# backbone_name = 'dino_vits16'

import timm
image_size = 224
num_channels = 3
backbone = timm.create_model('vit_small_patch16_224',
                             num_classes=0,
                             global_pool="token",
                             img_size=image_size,
                             dynamic_img_size=True,
                             in_chans=num_channels,
                             pretrained=False)

torchinfo.summary(backbone)
input_dim = backbone.embed_dim
batch_size = 32
backbone_name = 'vit_small_patch16_224'

ssl = DINOv1(backbone=backbone, input_dim=input_dim, max_epochs=max_epochs)

# DATA
num_workers = 16
transform = DINOTransformLightly(
        cj_prob = 0.0,
        random_gray_scale = 0,
        solarization_prob = 0,
)


def target_transform(t):
    return 0


ds = VOCDetection(transform=transform, target_transform=target_transform)
dl = torch.utils.data.DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
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
    'dataset': 'vocdetection',
    'backbone': backbone_name,
}

# TODO: this is overwritten by wandb.init
save_dir = '/users/amarti51/prometex/data/mae/logs'
wandb_logger = WandbLogger(project='dinov1',
                           name='dinov1',
                           config=config,
                           log_model=False,
                           save_dir=save_dir)

# CALLBACKS
lr_monitor = LearningRateMonitor(logging_interval="epoch")

# TRAINER
trainer = L.Trainer(
    max_epochs=max_epochs,
    logger=wandb_logger,
    callbacks=[lr_monitor],
    fast_dev_run=fast_dev_run,
)

trainer.fit(model=ssl, train_dataloaders=dl)
wandb.finish()

# %%
