# %%
from dataclasses import asdict

import lightning as L
import torch
import wandb
from lightning import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import Subset

from ai4bmr_learn.data_models.lightning_training import ProjectConfig, TrainerConfig, TrainingConfig, WandbInitConfig
from ai4bmr_learn.datasets.cifar10 import CIFAR10
from torchvision.transforms import v2
from ai4bmr_learn.models.backbones.base_backbone import BaseBackbone
from ai4bmr_learn.transforms.dino_transform import DINOTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE

# %% TRANSFORMS
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((224, 224)),
    v2.Normalize(mean=IMAGENET_NORMALIZE['mean'], std=IMAGENET_NORMALIZE['std'])
])

dino_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    DINOTransform(),
    v2.Normalize(mean=IMAGENET_NORMALIZE['mean'], std=IMAGENET_NORMALIZE['std'])
])

# %% DATA
random_indices = torch.randperm(50000)[:5000]
ds_test = Subset(CIFAR10(transform=transform), indices=random_indices)
ds_train = Subset(CIFAR10(transform=dino_transform), indices=random_indices)
# i = ds_train[0]

# %%
dl_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

item = ds_train[0]
batch = next(iter(dl_train))

dl_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

# %% BACKBONE
image_size = 224
num_channels = 3

model_name = 'vit_small_patch16_224'
backbone = BaseBackbone.from_timm_vit(model_name=model_name, image_size=image_size, num_channels=num_channels, dynamic_img_size=True)

backbone(batch['local_views'][0]['image'])

# %% CONFIGURATIONS
project_cfg = ProjectConfig(name="dinov1-cords2024")
trainer_cfg = TrainerConfig(max_epochs=1000,
                            accumulate_grad_batches=32,
                            # precision=16,
                            gradient_clip_val=1,
                            fast_dev_run=True)
training_cfg = TrainingConfig()
wandb_cfg = WandbInitConfig(project=project_cfg.name)

# %% SSL
from ai4bmr_learn.ssl.dinov1 import DINOv1
from lightly.models.modules import DINOProjectionHead

hidden_dim: int = 512
bottleneck_dim: int = 64
output_dim: int = 2048
student_head = DINOProjectionHead(
    input_dim=backbone.tokenizer.dim,
    hidden_dim=hidden_dim,
    bottleneck_dim=bottleneck_dim,
    output_dim=output_dim, freeze_last_layer=1
)
teacher_head = DINOProjectionHead(
    input_dim=backbone.tokenizer.dim,
    hidden_dim=hidden_dim,
    bottleneck_dim=bottleneck_dim,
    output_dim=output_dim,
)
ssl = DINOv1(backbone=backbone, teacher_head=teacher_head, student_head=student_head)
batch = {'views': [{'image':torch.randn(1, 43, 224, 224)}] * 6}
ssl.training_step(batch, batch_idx=0)

# LOGGER
from ai4bmr_learn.utils.utils import setup_wandb_auth
from ai4bmr_learn.utils.stats import model_stats

model_stats_dict = {f'student_backbone_{k}': v for k, v in model_stats(ssl.student_backbone).items()}
model_stats_dict.update({f'student_head_{k}': v for k, v in model_stats(ssl.student_head).items()})
model_stats_dict.update({f'teacher_backbone_{k}': v for k, v in model_stats(ssl.teacher_backbone).items()})
model_stats_dict.update({f'teacher_head_{k}': v for k, v in model_stats(ssl.teacher_head).items()})

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
    save_top_k=3,
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