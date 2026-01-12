import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from glom import glom
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

from ai4bmr_learn.utils.pooling import pool


class RegressionLit(L.LightningModule):
    def __init__(
            self,
            backbone: nn.Module,
            batch_key: str | None = "image",
            target_key: str = "y",
            lr_head: float = 1e-3,
            lr_backbone: float = 1e-4,
            weight_decay: float = 0.01,
            eta: float = 0.0,
            schedule: str | None = None,
            max_epochs: int = 1000,
            num_warmup_epochs: int = 10,
            freeze_backbone: bool = False,
            pooling: str | None = None,
            loss: str = "mse",  # "mse" or "huber"
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

        self.lr_head = lr_head
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

        self.criterion = self.configure_loss(loss=loss)

        metrics = self.get_metrics()
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # OPTIM
        self.schedule = schedule
        self.eta = eta
        self.max_epochs = max_epochs
        self.num_warmup_epochs = num_warmup_epochs

        self.save_hyperparameters(ignore=["head", "backbone"])

    def configure_loss(self, loss: str) -> nn.Module:
        if loss == "mse":
            return nn.MSELoss()
        elif loss == "huber":
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss: {loss}")

    def get_metrics(self) -> MetricCollection:
        return MetricCollection(
            {
                "mae": MeanAbsoluteError(),
                "mse": MeanSquaredError(squared=True),
                "rmse": MeanSquaredError(squared=False),
                "r2": R2Score(),  # does not work with single batch
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

        y = y.unsqueeze(1) if y.ndim == 1 else y  # [B, 1]
        y = y.float()

        z = self.backbone(x)
        z = pool(z, strategy=self.pooling)

        y_hat = self.head(z)
        assert y_hat.ndim == 2 and y_hat.shape[1] == 1, f"Expected y_hat [B,1], got {tuple(y_hat.shape)}"

        loss = self.criterion(y_hat, y)
        return z.detach().cpu(), y_hat.detach().cpu(), y.detach().cpu(), loss

    def training_step(self, batch, batch_idx: int):
        z, y_hat, y, loss = self.shared_step(batch, batch_idx)
        batch_size = int(y.shape[0])

        self.log("loss/train", loss, on_step=True, on_epoch=True, batch_size=batch_size)

        # metrics
        self.train_metrics.update(y_hat, y)

        batch["loss"] = loss
        batch["y_hat"] = y_hat
        batch["y"] = y
        batch["z"] = z
        return batch

    def on_train_epoch_end(self) -> None:
        if not self.trainer.fast_dev_run:
            # NOTE: we log here to support metrics that are only defined for B > 1 (e.g., R2Score)
            self.log_dict(self.train_metrics)

    def validation_step(self, batch, batch_idx: int):
        z, y_hat, y, loss = self.shared_step(batch, batch_idx)
        batch_size = int(y.shape[0])

        self.log("loss/val", loss, on_step=False, on_epoch=True, batch_size=batch_size)

        self.valid_metrics.update(y_hat, y)

        batch["loss"] = loss
        batch["y_hat"] = y_hat
        batch["y"] = y
        batch["z"] = z
        return batch

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.fast_dev_run:
            self.log_dict(self.valid_metrics)

    def test_step(self, batch, batch_idx: int):
        z, y_hat, y, loss = self.shared_step(batch, batch_idx)
        batch_size = int(y.shape[0])

        self.log("loss/test", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.test_metrics.update(y_hat, y)

        batch["loss"] = loss
        batch["y_hat"] = y_hat
        batch["y"] = y
        batch["z"] = z
        return batch

    def on_test_epoch_end(self) -> None:
        if not self.trainer.fast_dev_run:
            self.log_dict(self.test_metrics)

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
        optimizer = optim.AdamW([
            {'params': self.head.parameters(), 'lr': self.lr_head},
            {'params': filter(lambda p: p.requires_grad, self.backbone.parameters()), 'lr': self.lr_backbone}
        ], weight_decay=self.weight_decay)

        if self.schedule is None:
            return optimizer

        try:
            max_epochs = self.trainer.max_epochs
        except AttributeError:
            max_epochs = self.max_epochs

        num_warmup_epochs = self.num_warmup_epochs
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=num_warmup_epochs,
        )

        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - num_warmup_epochs,
            eta_min=self.eta,
        )

        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[num_warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
