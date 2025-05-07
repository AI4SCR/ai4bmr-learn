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
    def __init__(self, backbone: nn.Module | None = None, view_key: str = 'image',
                 input_dim: int = 512, hidden_dim: int = 512, bottleneck_dim: int = 64, output_dim: int = 2048):
        super().__init__()

        self.view_key = view_key

        if backbone is None:
            import torchvision
            resnet = torchvision.models.resnet18()
            backbone = nn.Sequential(*list(resnet.children())[:-1])

        # instead of a resnet you can also use a vision transformer backbone as in the
        # original paper (you might have to reduce the batch size in this case):
        # backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
        # input_dim = backbone.embed_dim

        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, hidden_dim, bottleneck_dim, output_dim, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, hidden_dim, bottleneck_dim, output_dim)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=output_dim, warmup_teacher_temp_epochs=5)

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
        batch_size = len(views[0][self.view_key])

        global_views = views[:2]
        local_views = views[2:]

        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)

        global_views = [view[self.view_key].to(self.device) for view in global_views]
        local_views = [view[self.view_key].to(self.device) for view in local_views]

        assert all(torch.isfinite(out).all() for out in global_views)
        assert all(torch.isfinite(out).all() for out in local_views)

        view = global_views[0]
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
