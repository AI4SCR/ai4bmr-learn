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
    def __init__(self, backbone: BaseBackbone, student_head: DINOProjectionHead, teacher_head: DINOProjectionHead, pooling: str = 'cls'):
        super().__init__()

        self.pooling = pooling

        self.student_backbone = backbone
        self.student_head = student_head

        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = teacher_head
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        output_dim = student_head.last_layer.out_features
        self.criterion = DINOLoss(output_dim=output_dim, warmup_teacher_temp_epochs=5)

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

    def training_step(self, batch, batch_idx):
        global_views = batch['global_views']
        local_views = batch['local_views']

        batch_size = len(local_views[0]['image'])

        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)

        teacher_out = [self.forward_teacher(view['image']) for view in global_views]
        student_out = [self.forward_student(view['image']) for view in local_views]

        assert all(torch.isfinite(out).all() for out in student_out)
        assert all(torch.isfinite(out).all() for out in teacher_out)

        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)

        assert not torch.isnan(loss).any()
        assert torch.isfinite(loss).all()

        self.log(
            "train_loss_epoch",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():

            global_views = batch['global_views']
            local_views = batch['local_views']

            batch_size = len(local_views[0]['image'])

            teacher_out = [self.forward_teacher(view['image']) for view in global_views]
            student_out = [self.forward_student(view['image']) for view in local_views]

            assert all(torch.isfinite(out).all() for out in teacher_out)
            assert all(torch.isfinite(out).all() for out in student_out)

            loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)

            assert not torch.isnan(loss).any()
            assert torch.isfinite(loss).all()

            self.log(
                "val_loss_epoch",
                loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

            return loss

    def predict_step(self, batch, batch_idx):
        x = self.student_backbone(batch)
        batch['embedding'] = x
        return batch

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return optim
