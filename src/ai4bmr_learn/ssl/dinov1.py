# https://github.com/google-research/vision_transformer?tab=readme-ov-file
# https://arxiv.org/abs/2106.10270
import copy

import lightning as L
import torch
import torch.nn as nn
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

def get_params_with_gradient(model: nn.Module):
    """
    Returns a list of parameters that require gradients in the model.
    """
    return [p for p in model.parameters() if p.requires_grad]


class DINOv1(L.LightningModule):
    name: str = 'DINOv1'

    def __init__(self,
                 backbone: nn.Module,
                 input_dim: int,
                 dino_head_output_dim: int = 2048,
                 dino_head_hidden_dim: int = 512,
                 dino_head_bottleneck_dim: int = 64,
                 lr: float = 5e-4,
                 global_batch_size: int = 64,
                 max_epochs: int = 150,
                 warmup_lr_epochs: int = 5,
                 weight_decay: float = 0.04,  # 0.01-0.02 for small, default: 0.04
                 betas: tuple[float, float] = (0.9, 0.999),  # (0.9, 0.95)
                 teacher_temp: float = 0.04,  # 0.06 – 0.07
                 warmup_teacher_temp_epochs: int = 3,
                 momentum: float = 0.996,  # 0.999
                 momentum_end: float = 1,
                 freeze_last_layer: int = 1,
                 norm_last_layer: bool = True,
                 pooling: str | None = 'flatten',
                 ):
        super().__init__()

        self.input_dim = input_dim

        self.max_epochs = max_epochs

        # lr = self.base_learning_rate * self.effective_batch_size / 256,
        self.lr = lr * global_batch_size / 256
        self.global_batch_size = global_batch_size
        self.warmup_lr_epochs = warmup_lr_epochs
        self.weight_decay = weight_decay

        self.betas = betas

        self.momentum = momentum
        self.momentum_end = momentum_end

        self.pooling = pooling

        # STUDENT
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(input_dim,
                                               dino_head_hidden_dim, dino_head_bottleneck_dim, dino_head_output_dim,
                                               freeze_last_layer=freeze_last_layer, norm_last_layer=norm_last_layer)
        # TEACHER
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim,
                                               dino_head_hidden_dim, dino_head_bottleneck_dim, dino_head_output_dim)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=dino_head_output_dim,
                                  warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
                                  teacher_temp=teacher_temp,
                                  )

        self.save_hyperparameters(ignore=['backbone', 'student_head', 'teacher_head'])

    def pool(self, x):
        if self.pooling is None:
            return x
        elif self.pooling == 'cls':
            return x[:, 0]
        elif self.pooling == 'flatten':
            return x.flatten(start_dim=1)
        else:
            raise NotImplementedError(f'{self.pooling} is not implemented.')

    def forward(self, x):
        y = self.student_backbone(x)
        y = self.pool(y)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x)
        y = self.pool(y)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        batch_size = batch['global_views'][0]['image'].shape[0]

        momentum = cosine_schedule(self.current_epoch, self.max_epochs, self.momentum, self.momentum_end)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)

        views = batch['global_views'] + batch['local_views']
        global_views = batch['global_views']

        teacher_out = [self.forward_teacher(view['image']) for view in global_views]
        student_out = [self.forward(view['image']) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)

        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('momentum', momentum, on_step=True, on_epoch=False, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        y = self.student_backbone(batch['image'])
        y = self.pool(y).cpu()

        batch['loss'] = 0
        batch['embedding'] = y
        return batch

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def predict_step(self, batch, batch_idx):
        y = self.student_backbone(batch['image'])
        y = self.pool(y).cpu()

        batch['embedding'] = y
        return batch

    def configure_optimizers(self):
        student_params = (list(self.student_backbone.parameters()) + list(self.student_head.parameters()))

        optimizer = torch.optim.AdamW(
            student_params,
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
            # https://towardsdatascience.com/weight-decay-and-its-peculiar-effects-66e0aee3e7b8/
            # eps=1e-3
        )

        # 1) Warm up from lr*1e-6 → lr over warmup_lr_epochs
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_lr_epochs,
        )
        # 2) Cosine decay from lr → 0 over (max_epochs - warmup_lr_epochs) epochs
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs - self.warmup_lr_epochs,
            eta_min=0.0,
        )
        # Chain them together at the warmup boundary
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_lr_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
