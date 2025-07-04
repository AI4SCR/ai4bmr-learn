"""
Self‑supervised ViT module.
"""

import copy
import math
from typing import List

import lightning as L
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm import create_model

from ai4bmr_learn.losses.dino_loss import DINOLoss
from ai4bmr_learn.losses.ibot import IBOTLoss


###############################################################################
#                   Utility helpers (no Lightly dependency)                   #
###############################################################################

def momentum_update(student: nn.Module, teacher: nn.Module, m: float) -> None:
    """EMA update of *teacher* parameters."""
    with torch.no_grad():
        for s, t in zip(student.parameters(), teacher.parameters()):
            t.data.mul_(m).add_(s.data, alpha=1.0 - m)


def cosine_scheduler(start: float, end: float, t: int, T: int) -> float:
    if T <= 0:
        return end
    cos_out = 0.5 * (1 + math.cos(math.pi * t / T))
    return end - (end - start) * cos_out


###############################################################################
#                            Projection Head                                  #
###############################################################################

class DINOProjectionHead(nn.Module):
    """Three‑layer projection head with weight‑norm last layer."""

    def __init__(self, in_dim: int, hidden_dim: int = 4096, bottleneck_dim: int = 256, nlayers: int = 3):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (nlayers - 1) + [bottleneck_dim]
        layers = []
        for i in range(nlayers):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            if i < nlayers - 1:
                layers.append(nn.GELU())
                layers.append(nn.BatchNorm1d(dims[i + 1]))
        self.mlp = nn.Sequential(*layers)

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, bottleneck_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return nn.functional.normalize(self.last_layer(x), dim=-1)

###############################################################################
#                               DINO Module                                   #
###############################################################################

class DINOv2(L.LightningModule):
    """ViT student‑teacher model with configurable self‑supervised loss."""

    def __init__(
        self,
        backbone_name: str = "vit_small_patch16_224",
        loss_type: str = "dino",  # "dino" | "ibot"
        num_global_views: int = 2,
        num_local_views: int = 6,
        max_epochs: int = 100,
        base_lr: float = 1e-3,
        weight_decay: float = 0.04,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # ---------------- Backbones ----------------
        self.student_backbone = create_model(backbone_name, pretrained=False, num_classes=0)
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False

        embed_dim = self.student_backbone.num_features
        self.student_head = DINOProjectionHead(embed_dim)
        self.teacher_head = copy.deepcopy(self.student_head)
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        # ---------------- Loss ----------------
        if loss_type == "dino":
            self.criterion = DINOLoss(out_dim=256, total_epochs=max_epochs)
        elif loss_type == "ibot":
            self.criterion = IBOTLoss(cls_dim=256, patch_dim=256, total_epochs=max_epochs)
        else:
            raise ValueError(f"Unknown loss_type {loss_type}")

        self.loss_type = loss_type
        self.num_global_views = num_global_views
        self.num_local_views = num_local_views

    # -----------------------------------------------------------
    def _forward_backbone(self, x, backbone, head, return_patches=False):
        y = backbone(x)  # ViT: (bs, 1+N, dim)
        if y.dim() == 3:
            cls = y[:, 0]
            patches = y[:, 1:]
        else:  # ConvNets etc.
            cls = y
            patches = None
        cls_feat = head(cls)
        if return_patches and patches is not None:
            patch_feat = head(patches)  # apply projection token‑wise via broadcast
            return cls_feat, patch_feat
        return cls_feat

    # -----------------------------------------------------------
    def _shared_step(self, batch, _):
        g_views: List[torch.Tensor] = batch["global_views"]
        l_views: List[torch.Tensor] = batch["local_views"]
        mask: torch.Tensor = batch.get("mask", None)  # (bs, n_patches) for iBOT

        m = cosine_scheduler(0.996, 1.0, self.current_epoch, self.hparams.max_epochs)
        momentum_update(self.student_backbone, self.teacher_backbone, m)
        momentum_update(self.student_head, self.teacher_head, m)

        if self.loss_type == "dino":
            with torch.no_grad():
                teacher_out = [
                    self._forward_backbone(v, self.teacher_backbone, self.teacher_head)
                    for v in g_views
                ]
            student_out = [
                self._forward_backbone(v, self.student_backbone, self.student_head)
                for v in g_views + l_views
            ]
            loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)

        else:  # iBOT
            with torch.no_grad():
                teacher_cls, teacher_patch = zip(*[
                    self._forward_backbone(v, self.teacher_backbone, self.teacher_head, return_patches=True)
                    for v in g_views
                ])
            student_cls, student_patch = zip(*[
                self._forward_backbone(v, self.student_backbone, self.student_head, return_patches=True)
                for v in g_views + l_views
            ])
            loss = self.criterion(
                teacher_cls_out=list(teacher_cls),
                student_cls_out=list(student_cls),
                teacher_patch_out=list(teacher_patch),
                student_patch_out=list(student_patch),
                mask=mask,
                epoch=self.current_epoch,
            )
        return loss, g_views[0].size(0)

    # -----------------------------------------------------------
    def training_step(self, batch, batch_idx):
        loss, bs = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, bs = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, batch_size=bs)

    def on_after_backward(self):
        if hasattr(self.student_head.last_layer, "weight_g"):
            self.student_head.last_layer.weight_g.grad = None

    def configure_optimizers(self):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n.endswith("bias") or n.endswith("weight_g") or "norm" in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        opt = AdamW([
            {"params": decay, "weight_decay": self.hparams.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=self.hparams.base_lr, betas=(0.9, 0.999))
        sched = CosineAnnealingLR(opt, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": sched}

    def on_train_start(self):
        if self.trainer.precision != 16:
            self.trainer.precision = 16
