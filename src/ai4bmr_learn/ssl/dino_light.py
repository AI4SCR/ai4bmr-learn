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
from torch import nn


class DINOLight(L.LightningModule):
    def __init__(self, backbone: nn.Module | None = None, input_dim: int | None = None):
        super().__init__()

        if backbone is None:
            import torchvision
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
        views = batch['views']
        batch_size = len(views[0]['image'])

        global_views = views[:2]
        local_views = views[2:]

        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)

        global_views = [view['image'].to(self.device) for view in global_views]
        local_views = [view['image'].to(self.device) for view in local_views]

        assert all(torch.isfinite(out).all() for out in global_views)
        assert all(torch.isfinite(out).all() for out in local_views)

        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in local_views]

        assert all(torch.isfinite(out).all() for out in student_out)
        assert all(torch.isfinite(out).all() for out in teacher_out)

        assert not any(torch.isnan(out).any() for out in student_out)
        assert not any(torch.isnan(out).any() for out in teacher_out)

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

    def predict_step(self, batch, batch_idx):
        x = batch['image']
        x = self.student_backbone(x)
        batch['embedding'] = x
        return batch

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return optim
