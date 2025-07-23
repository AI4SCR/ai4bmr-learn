#!/usr/bin/env python3
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from ai4bmr_learn.callbacks.LinearProbing import LinearProbing
from ai4bmr_learn.callbacks.UMAP import UMAP
from ai4bmr_learn.datamodules.prepare_dataset import Cords2024
from ai4bmr_learn.models.backbones.timm import Backbone
from ai4bmr_learn.ssl.dinov1 import DINOv1

# reproducibility
seed_everything(0)

# logger
wandb_logger = WandbLogger(
    project="dinov1-cords2024",
    save_dir="/work/FAC/FBM/DBC/mrapsoma/prometex/data/dinov1/logs",
    log_model=False
)

# callbacks
lr_monitor = LearningRateMonitor(logging_interval="epoch")
checkpoint_cb = ModelCheckpoint(
    monitor="linear_probing/balanced_accuracy",
    save_last=True
)
linear_probe_cb = LinearProbing(
    target_key="clinical.dx_name",
    run_before_train=True,
    run_every_num_epochs=3
)
umap_cb = UMAP(
    label_key="clinical.dx_name",
    log_before_train=True,
    log_every_num_epochs=3
)

# data
dm = Cords2024(
    batch_size=64,
    num_workers=16,
    persistent_workers=False,
    shuffle=True,
    pin_memory=False
)

# model backbone
backbone = Backbone(
    model_name="vit_small_patch16_224",
    num_channels=43
)

# full LightningModule
model = DINOv1(
    backbone=backbone,
    input_dim=384,
    max_epochs=1500,
    lr=5e-4,
    global_batch_size=64,
    warmup_lr_epochs=5,
    weight_decay=0.04,
    betas=(0.9, 0.999),
    teacher_temp=0.04,
    warmup_teacher_temp_epochs=3,
    momentum=0.996,
    momentum_end=1,
    freeze_last_layer=1,
    norm_last_layer=True,
    pooling="flatten"
)

# model stats
import torchinfo
import torch
import torch.nn as nn
# inp = torch.randn((8, 43, 224, 224))
model = nn.Sequential([DINOv1.student_backbone])
torchinfo.summary(model, input_size=(64, 43, 224, 224))

# trainer
trainer = pl.Trainer(
    accelerator="auto",
    strategy="auto",
    devices="auto",
    num_nodes=1,
    precision=None,
    logger=wandb_logger,
    callbacks=[lr_monitor, checkpoint_cb, linear_probe_cb, umap_cb],
    fast_dev_run=False,
    max_epochs=1500,
    max_time="00:23:55:00",
    check_val_every_n_epoch=1,
    accumulate_grad_batches=1,
)

# run!
trainer.fit(model, datamodule=dm)
