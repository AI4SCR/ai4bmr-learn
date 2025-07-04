# This example requires the following dependencies to be installed:
# pip install lightly

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import copy
import wandb
import lightning as L
import torch
import torchvision
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule
from torch import nn

max_epochs = 300
num_workers = 8
batch_size = 64

class DINO(L.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        input_dim = 512
        # instead of a resnet you can also use a vision transformer backbone as in the
        # original paper (you might have to reduce the batch size in this case):
        # backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
        # input_dim = backbone.embed_dim

        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, max_epochs, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        views = batch[0]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)

        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True)
        self.log('momentum', momentum, on_step=True, on_epoch=False)

        return loss

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return optim


model = DINO()

transform = DINOTransform()
def target_transform(t):
    return 0

dataset = torchvision.datasets.VOCDetection(
    "/users/amarti51/prometex/data/datasets/pascal_voc",
    download=True,
    transform=transform,
    target_transform=target_transform,
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)

accelerator = "gpu" if torch.cuda.is_available() else "cpu"

# %%
from dotenv import load_dotenv

load_dotenv()

import lightning as L
import torch
from ai4bmr_learn.utils.utils import setup_wandb_auth
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from ai4bmr_learn.data_models.lightning_training import ProjectConfig, WandbInitConfig

project_cfg = ProjectConfig(name="dinov1-cifar10")
wandb_cfg = WandbInitConfig(project=project_cfg.name)
setup_wandb_auth()
wandb_logger = WandbLogger(project=project_cfg.name, name='lightly', log_model=False, save_dir=project_cfg.log_dir)

# CALLBACKS
lr_monitor = LearningRateMonitor(logging_interval="epoch")

trainer = L.Trainer(
    max_epochs=max_epochs,
    accelerator=accelerator,
    logger=wandb_logger,
    callbacks=[lr_monitor],
)
trainer.fit(model=model, train_dataloaders=dataloader)
wandb.finish()
