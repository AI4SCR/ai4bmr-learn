# %%
from dataclasses import asdict

from dotenv import load_dotenv

load_dotenv()

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from ai4bmr_learn.data_models.lightning_training import ProjectConfig, WandbInitConfig
from ai4bmr_learn.datasets.vocdetection import VOCDetection

from ai4bmr_learn.transforms.dino_transform import DINOTransformLightly

# DATA
batch_size = 64
num_workers = 14

transform = DINOTransformLightly()
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

# %% CONFIGURATIONS
project_cfg = ProjectConfig(name="dinov1-cifar10")
wandb_cfg = WandbInitConfig(project=project_cfg.name, name='dinov1')

# %% SSL
from ai4bmr_learn.ssl.dinov1 import DINOv1
max_epochs = 300
ssl = DINOv1(max_epochs=max_epochs)

# LOGGER
from ai4bmr_learn.utils.utils import setup_wandb_auth
from ai4bmr_learn.utils.stats import model_stats

model_stats_dict = {f'student_backbone_{k}': v for k, v in model_stats(ssl.student_backbone).items()}
model_stats_dict.update({f'student_head_{k}': v for k, v in model_stats(ssl.student_head).items()})
model_stats_dict.update({f'teacher_backbone_{k}': v for k, v in model_stats(ssl.teacher_backbone).items()})
model_stats_dict.update({f'teacher_head_{k}': v for k, v in model_stats(ssl.teacher_head).items()})

setup_wandb_auth()

metadata = {
    'dataset': 'vocdetection',
    'backbone': 'resnet18',
}

fast_dev_run = False
if not fast_dev_run:
    import wandb

    wandb.init(**asdict(wandb_cfg))
    ckpt_dir = project_cfg.ckpt_dir / wandb.run.name
    metadata["ckpt_dir"] = ckpt_dir
    wandb.config.update(metadata)
else:
    ckpt_dir = None

# TODO: this is overwritten by wandb.init
wandb_logger = WandbLogger(project=project_cfg.name, name='dinov1',
                           log_model=False, save_dir=project_cfg.log_dir)

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
