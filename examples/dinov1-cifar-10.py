# %%
from dataclasses import asdict
from dotenv import load_dotenv

load_dotenv()

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
from ai4bmr_learn.transforms.dino_transform import DINOTransform, DINOTransformLightly
from lightly.transforms.utils import IMAGENET_NORMALIZE

# TRANSFORMS
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

dino_transform_lightly = DINOTransformLightly(local_crop_scale=(0.15, 0.25),
                                              # cj_prob=0.4,
                                              # cj_strength=0.0,
                                              # cj_bright=0.0,
                                              # cj_contrast=0.0,
                                              # cj_sat=0.,
                                              # cj_hue=0.,
                                              # solarization_prob=0,
                                              normalize=None
                                              )

# DATA
random_indices = torch.randperm(50000)[:15000]


ds_test = Subset(CIFAR10(transform=transform), indices=random_indices)
ds_train = Subset(CIFAR10(transform=dino_transform_lightly), indices=random_indices[:12000])
ds_val = Subset(CIFAR10(transform=dino_transform_lightly), indices=random_indices[12000:])


from matplotlib import pyplot as plt
from torchvision.utils import make_grid

ds = CIFAR10(transform=None)
item = ds[0]
plt.imshow(torch.tensor(item['image'])).figure.show()
grid = make_grid([i['image'] for i in dino_transform_lightly(item)['local_views']])
plt.imshow(grid.permute((1, 2, 0))).figure.show()

grid = make_grid([i['image'] for i in dino_transform_lightly(item)['global_views']])
plt.imshow(grid.permute((1, 2, 0))).figure.show()

# DATA LOADERS
num_workers = 14
batch_size = 64

dl_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
)

dl_val = torch.utils.data.DataLoader(
    ds_val,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
)

dl_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
)

# BACKBONE
image_size = 224
num_channels = 3

model_name = 'vit_small_patch16_224'
# model_name = 'vit_base_patch16_224'
backbone = BaseBackbone.from_timm_vit(model_name=model_name, image_size=image_size, num_channels=num_channels,
                                      dynamic_img_size=True)

# CONFIGURATIONS
project_cfg = ProjectConfig(name="dinov1-cifar10")
trainer_cfg = TrainerConfig(max_epochs=300,
                            accumulate_grad_batches= 512 // batch_size,
                            # precision='16-mixed',
                            gradient_clip_val=3.0,
                            fast_dev_run=False)
training_cfg = TrainingConfig()
wandb_cfg = WandbInitConfig(project=project_cfg.name)

# %% SSL
from ai4bmr_learn.ssl.dinov1 import DINOv1
from lightly.models.modules import DINOProjectionHead

hidden_dim, = backbone.encoder.model.norm.normalized_shape
bottleneck_dim: int = 64 * 2
output_dim: int = 2048 * 2
student_head = DINOProjectionHead(
    input_dim=backbone.tokenizer.dim,
    hidden_dim=hidden_dim,
    bottleneck_dim=bottleneck_dim,
    output_dim=output_dim,
    freeze_last_layer=3
)
teacher_head = DINOProjectionHead(
    input_dim=backbone.tokenizer.dim,
    hidden_dim=hidden_dim,
    bottleneck_dim=bottleneck_dim,
    output_dim=output_dim,
)
ssl = DINOv1(backbone=backbone,
             teacher_head=teacher_head,
             student_head=student_head,
             batch_size=batch_size,
             accumulate_grad_batches=trainer_cfg.accumulate_grad_batches,
             epochs=trainer_cfg.max_epochs
             )

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
trainer.fit(model=ssl, train_dataloaders=dl_train, ckpt_path=ckpt_path)

# Finish the wandb run
wandb.finish()

# %%
