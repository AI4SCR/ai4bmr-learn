from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from glom import glom
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

from ai4bmr_learn.utils.pooling import pool

from torchmetrics import Metric
from torchsurv.loss import cox


class ConcordanceIndex(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("events", default=[], dist_reduce_fx="cat")
        self.add_state("times", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, events: torch.Tensor, times: torch.Tensor) -> None:
        self.preds.append(preds.detach().cpu())
        self.events.append(events.detach().cpu())
        self.times.append(times.detach().cpu())

    def compute(self) -> torch.Tensor:
        from torchsurv.metrics.cindex import ConcordanceIndex

        preds = torch.cat(self.preds).numpy()
        events = torch.cat(self.events).numpy()
        times = torch.cat(self.times).numpy()

        res = ConcordanceIndex(preds, events, times)

        return torch.tensor(res, dtype=torch.float32)


class SurvivalLit(L.LightningModule):
    def __init__(
            self,
            backbone: nn.Module,
            batch_key: str | None = "image",
            target_key: str = "y",
            event_key: str = "event",
            lr_head: float = 1e-3,
            lr_backbone: float = 1e-4,
            weight_decay: float = 0.01,
            freeze_backbone: bool = False,
            pooling: str | None = None,
    ):
        super().__init__()

        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        input_dim = backbone.output_dim
        self.head = nn.Linear(input_dim, 1)

        self.pooling = pooling
        self.batch_key = batch_key
        self.target_key = target_key
        self.event_key = event_key

        self.lr_head = lr_head
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

        metrics = self.get_metrics()
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.train_cindex = ConcordanceIndex()
        self.valid_cindex = ConcordanceIndex()
        self.test_cindex = ConcordanceIndex()

        self.save_hyperparameters(ignore=["head", "backbone"])

    def get_metrics(self) -> MetricCollection:
        return MetricCollection(
            {
                "mae": MeanAbsoluteError(),
                "mse": MeanSquaredError(squared=True),
                "rmse": MeanSquaredError(squared=False),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        z = pool(z, strategy=self.pooling)
        y_hat = self.head(z)  # [B, 1]
        return y_hat

    def shared_step(self, batch: dict, batch_idx: int):
        x = glom(batch, self.batch_key) if self.batch_key is not None else batch
        y = glom(batch, self.target_key)
        events = glom(batch, self.event_key)

        y = y.unsqueeze(1) if y.ndim == 1 else y  # [B, 1]
        y = y.float()

        z = self.backbone(x)
        z = pool(z, strategy=self.pooling)

        y_hat = self.head(z)
        assert y_hat.ndim == 2 and y_hat.shape[1] == 1, f"Expected y_hat [B,1], got {tuple(y_hat.shape)}"

        loss = cox.neg_partial_log_likelihood(y_hat, events, y)
        return z, y_hat, y, events, loss

    def training_step(self, batch, batch_idx: int):
        z, y_hat, y, events, loss = self.shared_step(batch, batch_idx)
        batch_size = int(y.shape[0])

        self.log("loss/train", loss, on_step=True, on_epoch=True, batch_size=batch_size)

        # metrics
        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, batch_size=batch_size)

        self.train_cindex(y_hat, events, y)
        self.log_dict(self.train_cindex, on_step=False, on_epoch=True, batch_size=batch_size)

        batch["loss"] = loss
        batch["y_hat"] = y_hat.detach().cpu()
        batch["y"] = y.detach().cpu()
        batch["z"] = z.detach().cpu()
        return batch

    def validation_step(self, batch, batch_idx: int):
        z, y_hat, y, events, loss = self.shared_step(batch, batch_idx)
        batch_size = int(y.shape[0])

        self.log("loss/val", loss, on_step=False, on_epoch=True, batch_size=batch_size)

        self.valid_metrics(y_hat, y)
        self.log_dict(self.valid_metrics, on_step=False, on_epoch=True, batch_size=batch_size)

        self.valid_cindex(y_hat, events, y)
        self.log_dict(self.valid_cindex, on_step=False, on_epoch=True, batch_size=batch_size)

        batch["loss"] = loss
        batch["y_hat"] = y_hat.detach().cpu()
        batch["y"] = y.detach().cpu()
        batch["z"] = z.detach().cpu()
        return batch

    def test_step(self, batch, batch_idx: int):
        z, y_hat, y, events, loss = self.shared_step(batch, batch_idx)
        batch_size = int(y.shape[0])

        self.log("loss/test", loss, on_step=False, on_epoch=True, batch_size=batch_size)

        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, batch_size=batch_size)

        self.test_cindex(y_hat, events, y)
        self.log_dict(self.test_cindex, on_step=False, on_epoch=True, batch_size=batch_size)

        batch["loss"] = loss
        batch["y_hat"] = y_hat.detach().cpu()
        batch["y"] = y.detach().cpu()
        batch["z"] = z.detach().cpu()
        return batch

    def predict_step(self, batch, batch_idx: int):
        x = glom(batch, self.batch_key) if self.batch_key is not None else batch

        z = self.backbone(x)
        z = pool(z, strategy=self.pooling)
        y_hat = self.head(z)

        batch["prediction"] = y_hat.detach().cpu()  # [B,1]
        batch["y_hat"] = y_hat.detach().cpu()
        batch["z"] = z.detach().cpu()

        y = glom(batch, self.target_key)
        assert isinstance(y, torch.Tensor), f"Expected target tensor at '{self.target_key}', got {type(y)}"
        if y.ndim == 1:
            y = y.unsqueeze(1)
        batch["y"] = y.detach().cpu()

        return batch

    def configure_optimizers(self):
        optimizer = optim.Adam(
            [
                {"params": self.head.parameters(), "lr": self.lr_head},
                {
                    "params": filter(lambda p: p.requires_grad, self.backbone.parameters()),
                    "lr": self.lr_backbone,
                },
            ],
            weight_decay=self.weight_decay,
        )
        return optimizer
