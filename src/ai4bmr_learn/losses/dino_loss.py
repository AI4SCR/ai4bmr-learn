from typing import List

import torch
from torch import nn as nn


class DINOLoss(nn.Module):
    """DINO self‑distillation loss (cls‑token only)."""

    def __init__(
        self,
        out_dim: int,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.07,
        warmup_teacher_temp_epochs: int = 10,
        total_epochs: int = 100,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

        self.warmup_teacher_temp = warmup_teacher_temp
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.total_epochs = total_epochs

    def _teacher_temperature(self, epoch: int) -> float:
        if epoch < self.warmup_teacher_temp_epochs:
            return self.warmup_teacher_temp + (
                (self.teacher_temp - self.warmup_teacher_temp)
                * epoch
                / self.warmup_teacher_temp_epochs
            )
        return self.teacher_temp

    def forward(self, teacher_out: List[torch.Tensor], student_out: List[torch.Tensor], epoch: int) -> torch.Tensor:
        student_logits = torch.stack(student_out) / self.student_temp  # (n_s, bs, d)
        student_logprob = nn.functional.log_softmax(student_logits, dim=-1)

        teacher_temp = self._teacher_temperature(epoch)
        with torch.no_grad():
            teacher_logits = (torch.stack(teacher_out) - self.center) / teacher_temp
            teacher_prob = nn.functional.softmax(teacher_logits, dim=-1)
            batch_center = teacher_prob.mean(dim=(0, 1))
            self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        n_t, n_s = teacher_prob.size(0), student_logprob.size(0)
        loss = 0.0
        for it in range(n_t):
            loss += -(
                teacher_prob[it].detach()
                * student_logprob.sum(dim=0)
            ).sum(dim=-1).mean()
        loss /= n_t * n_s
        return loss
