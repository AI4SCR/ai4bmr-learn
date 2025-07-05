import copy

import lightning as L
import torch
import torch.nn as nn
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule


class DINOv1(L.LightningModule):


    def __init__(self, backbone: nn.Module, input_dim: int, max_epochs: int):
        super().__init__()

        self.backbone = backbone
        self.input_dim = 512
        self.max_epochs = max_epochs

        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)

        self.save_hyperparameters(ignore=['backbone', 'student_head', 'teacher_head'])

    # def pool(self, x):
    #     if self.pooling == 'cls':
    #         return x[:, 0]
    #     elif self.pooling == 'flatten':
    #         return x.flatten(start_dim=1)
    #     else:
    #         raise NotImplementedError(f'{self.pooling} is not implemented.')

    # def forward_student(self, x):
    #     x = self.student_backbone(x)
    #     x = self.pool(x)
    #     x = self.student_head(x)
    #     return x
    #
    # def forward_teacher(self, x):
    #     x = self.teacher_backbone(x)
    #     x = self.pool(x)
    #     x = self.teacher_head(x)
    #     return x

    # def shared_step(self, batch, batch_idx):
    #     global_views = batch['global_views']
    #     local_views = batch['local_views']
    #
    #     batch_size = len(local_views[0]['image'])
    #
    #     momentum = cosine_schedule(self.current_epoch, self.epochs, 0.996, 1)
    #     self.log('momentum', momentum, on_step=True, on_epoch=False)
    #
    #     update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
    #     update_momentum(self.student_head, self.teacher_head, m=momentum)
    #
    #     teacher_out = [self.forward_teacher(view['image']) for view in global_views]
    #     student_out = [self.forward_student(view['image']) for view in local_views]
    #
    #     try:
    #         assert all(torch.isfinite(out).all() for out in student_out)
    #     except AssertionError as e:
    #         print(f'student_out has na for {batch_idx}')
    #
    #         # img = global_views[0]['image']
    #         # c, *_ = torch.where(img.isnan())
    #         # sample_index = torch.unique(c).item()
    #         # batch['index'][sample_index]
    #         #
    #         # layer = img[57]
    #         # c, h, w = torch.where(layer.isnan())
    #         # torch.unique(c)
    #     try:
    #         assert all(torch.isfinite(out).all() for out in teacher_out)
    #     except AssertionError as e:
    #         print(f'teacher_out has na for {batch_idx}')
    #
    #     loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
    #
    #     try:
    #         assert not torch.isnan(loss).any()
    #         assert torch.isfinite(loss).all()
    #     except AssertionError as e:
    #         print(f'loss has na for {batch_idx}')
    #         loss = 0.1
    #
    #     return loss, batch_size

    # def training_step(self, batch, batch_idx):
    #     loss, batch_size = self.shared_step(batch, batch_idx=batch_idx)
    #
    #     self.log(
    #         "train_loss_epoch",
    #         loss,
    #         on_step=False,
    #         on_epoch=True,
    #         batch_size=batch_size,
    #     )
    #
    # def validation_step(self, batch, batch_idx):
    #     with torch.no_grad():
    #         loss, batch_size = self.shared_step(batch, batch_idx=batch_idx)
    #
    #     self.log(
    #         "val_loss_epoch",
    #         loss,
    #         on_step=False,
    #         on_epoch=True,
    #         batch_size=batch_size,
    #     )
    #
    # def predict_step(self, batch, batch_idx):
    #     x = self.student_backbone(batch)
    #     batch['embedding'] = x
    #     return batch

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, self.max_epochs, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)

        views = batch['global_views'] + batch['local_views']
        global_views = batch['global_views']

        teacher_out = [self.forward_teacher(view['image']) for view in global_views]
        student_out = [self.forward(view['image']) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)

        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True)
        self.log('momentum', momentum, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        y = self.student_backbone(batch['image']).flatten(start_dim=1).cpu()
        batch['loss'] = 0
        batch['embedding'] = y
        return batch

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return optim

    def predict_step(self, batch, batch_idx):
        y = self.student_backbone(batch['image']).flatten(start_dim=1)
        batch['embedding'] = y
        return batch