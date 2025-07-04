# This example requires the following dependencies to be installed:
# pip install lightly

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import copy

import lightning as L
import torch
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
from ai4bmr_learn.models.backbones.base_backbone import BaseBackbone


class DINOv1(L.LightningModule):
    def __init__(self,
                 backbone: BaseBackbone,
                 student_head: DINOProjectionHead,
                 teacher_head: DINOProjectionHead,
                 pooling: str = 'cls',
                 batch_size: int = 64,
                 accumulate_grad_batches: int = 8,
                 base_learning_rate=5e-4,
                 weight_decay: float = 0.1,
                 warmup_epochs: int = 10,
                 epochs: int = 300,
                 warmup_teacher_temp_epochs=30,
                 warmup_teacher_temp=0.04,
                 teacher_temp=0.06,
                 ):
        super().__init__()

        self.pooling = pooling

        self.student_backbone = backbone
        self.student_head = student_head

        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = teacher_head
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        output_dim = student_head.last_layer.out_features
        self.criterion = DINOLoss(output_dim=output_dim,
                                  warmup_teacher_temp=warmup_teacher_temp,
                                  warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
                                  teacher_temp=teacher_temp)

        # optimizer
        self.batch_size = batch_size
        self.accumulate_grad_batches = accumulate_grad_batches
        self.effective_batch_size = batch_size * accumulate_grad_batches
        self.base_learning_rate = base_learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs

        self.save_hyperparameters(ignore=['backbone', 'student_head', 'teacher_head'])

    def pool(self, x):
        if self.pooling == 'cls':
            return x[:, 0]
        else:
            raise NotImplementedError(f'{self.pooling} is not implemented.')

    def forward_student(self, x):
        x = self.student_backbone(x)
        x = self.pool(x)
        x = self.student_head(x)
        return x

    def forward_teacher(self, x):
        x = self.teacher_backbone(x)
        x = self.pool(x)
        x = self.teacher_head(x)
        return x

    def shared_step(self, batch, batch_idx):
        global_views = batch['global_views']
        local_views = batch['local_views']

        batch_size = len(local_views[0]['image'])

        momentum = cosine_schedule(self.current_epoch, self.epochs, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)

        teacher_out = [self.forward_teacher(view['image']) for view in global_views]
        student_out = [self.forward_student(view['image']) for view in local_views]

        try:
            assert all(torch.isfinite(out).all() for out in student_out)
        except AssertionError as e:
            print(f'student_out has na for {batch_idx}')

            # img = global_views[0]['image']
            # c, *_ = torch.where(img.isnan())
            # sample_index = torch.unique(c).item()
            # batch['index'][sample_index]
            #
            # layer = img[57]
            # c, h, w = torch.where(layer.isnan())
            # torch.unique(c)



        try:
            assert all(torch.isfinite(out).all() for out in teacher_out)
        except AssertionError as e:
            print(f'teacher_out has na for {batch_idx}')

        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)

        try:
            assert not torch.isnan(loss).any()
            assert torch.isfinite(loss).all()
        except AssertionError as e:
            print(f'loss has na for {batch_idx}')
            loss = 0.1

        return loss, batch_size

    def training_step(self, batch, batch_idx):
        loss, batch_size = self.shared_step(batch, batch_idx=batch_idx)

        self.log(
            "train_loss_epoch",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, batch_size = self.shared_step(batch, batch_idx=batch_idx)

        self.log(
            "val_loss_epoch",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

    def predict_step(self, batch, batch_idx):
        x = self.student_backbone(batch)
        batch['embedding'] = x
        return batch

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        import math

        # https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.base_learning_rate * self.effective_batch_size / 256,
            betas=(0.9, 0.95),
            weight_decay=self.weight_decay,
        )
        lr_func = lambda epoch: min(
            (epoch + 1) / (self.warmup_epochs + 1e-8),
            0.5 * (math.cos(epoch / self.epochs * math.pi) + 1),
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
