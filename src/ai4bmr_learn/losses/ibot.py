from typing import List

import torch
from torch import nn as nn


class IBOTLoss(nn.Module):
    """iBOT combined cls‑token **+** masked‑patch distillation loss.

    Parameters
    ----------
    cls_dim, patch_dim: embedding dimensions of cls and patch heads.
    lambda_patch: weight assigned to patch loss (paper uses 1.0).
    mask_value: boolean mask selects which patches are *masked* for student.
    """

    def __init__(
        self,
        cls_dim: int,
        patch_dim: int,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.07,
        warmup_teacher_temp_epochs: int = 10,
        total_epochs: int = 100,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        lambda_patch: float = 1.0,
    ) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.lambda_patch = lambda_patch
        self.center_momentum = center_momentum

        # Two independent centers (cls & patch)
        self.register_buffer("center_cls", torch.zeros(1, cls_dim))
        self.register_buffer("center_patch", torch.zeros(1, patch_dim))

        self.warmup_teacher_temp = warmup_teacher_temp
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.total_epochs = total_epochs

    # ------------------------- helpers ------------------------- #
    def _teacher_temperature(self, epoch: int) -> float:
        if epoch < self.warmup_teacher_temp_epochs:
            return self.warmup_teacher_temp + (
                (self.teacher_temp - self.warmup_teacher_temp)
                * epoch
                / self.warmup_teacher_temp_epochs
            )
        return self.teacher_temp

    # ------------------------- forward ------------------------- #
    def forward(
        self,
        teacher_cls_out: List[torch.Tensor],       # list of (bs, cls_dim)
        student_cls_out: List[torch.Tensor],       # list of (bs, cls_dim)
        teacher_patch_out: List[torch.Tensor],     # list of (bs, n_patches, patch_dim)
        student_patch_out: List[torch.Tensor],     # list of (bs, n_patches, patch_dim)
        mask: torch.Tensor,                        # (bs, n_patches) bool — True if **masked**
        epoch: int,
    ) -> torch.Tensor:
        """Compute the sum of cls‑token DINO loss and masked‑patch distillation."""
        # ---------------- CLS loss (same as DINO) -----------------
        student_cls_logits = torch.stack(student_cls_out) / self.student_temp
        student_cls_logprob = nn.functional.log_softmax(student_cls_logits, dim=-1)

        temp = self._teacher_temperature(epoch)
        with torch.no_grad():
            teacher_cls_logits = (torch.stack(teacher_cls_out) - self.center_cls) / temp
            teacher_cls_prob = nn.functional.softmax(teacher_cls_logits, dim=-1)
            new_center_cls = teacher_cls_prob.mean(dim=(0, 1))
            self.center_cls = self.center_cls * self.center_momentum + new_center_cls * (1 - self.center_momentum)

        n_t, n_s = teacher_cls_prob.size(0), student_cls_logprob.size(0)
        cls_loss = 0.0
        for it in range(n_t):
            cls_loss += -(
                teacher_cls_prob[it].detach()
                * student_cls_logprob.sum(dim=0)
            ).sum(dim=-1).mean()
        cls_loss /= n_t * n_s

        # ---------------- Patch loss (masked only) ---------------
        # stack shapes: (n_views, bs, n_patches, dim)
        student_patch_logits = torch.stack(student_patch_out) / self.student_temp
        student_patch_logprob = nn.functional.log_softmax(student_patch_logits, dim=-1)

        with torch.no_grad():
            teacher_patch_logits = (torch.stack(teacher_patch_out) - self.center_patch) / temp
            teacher_patch_prob = nn.functional.softmax(teacher_patch_logits, dim=-1)
            new_center_patch = teacher_patch_prob.mean(dim=(0, 1, 2))  # over views, bs, patches
            self.center_patch = self.center_patch * self.center_momentum + new_center_patch * (1 - self.center_momentum)

        # Ensure mask broadcast: (1, bs, n_patches, 1)
        mask_broadcast = mask.unsqueeze(0).unsqueeze(-1)  # bool
        teacher_masked = teacher_patch_prob.detach()[mask_broadcast].view(-1, teacher_patch_prob.size(-1))
        student_masked = student_patch_logprob[mask_broadcast].view(-1, student_patch_logprob.size(-1))

        patch_loss = -(teacher_masked * student_masked).sum(dim=-1).mean()

        # ---------------- Total ----------------
        return cls_loss + self.lambda_patch * patch_loss
