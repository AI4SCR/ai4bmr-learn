import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from glom import glom
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score, SpearmanCorrCoef, PearsonCorrCoef

from ai4bmr_learn.utils.pooling import pool


class RegressionLit(L.LightningModule):
    def __init__(
            self,
            backbone: nn.Module,
            head: nn.Module | None = None,
            embed_dim: int | None = None,
            num_outputs: int = 1,
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
            save_hparams: bool = True,
    ):
        super().__init__()

        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if head is None:
            input_dim = embed_dim or backbone.output_dim
            head = nn.Linear(input_dim, 1)

        self.head = head

        self.pooling = pooling
        self.batch_key = batch_key
        self.target_key = target_key

        self.lr_head = lr_head
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

        self.criterion = self.configure_loss(loss=loss)

        self.num_outputs = num_outputs
        metrics = self.get_metrics(num_outputs=num_outputs)
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # OPTIM
        self.schedule = schedule
        self.eta = eta
        self.max_epochs = max_epochs
        self.num_warmup_epochs = num_warmup_epochs

        if save_hparams:
            self.save_hyperparameters(ignore=["head", "backbone"])

    def configure_loss(self, loss: str) -> nn.Module:
        if loss == "mse":
            return nn.MSELoss()
        elif loss == "huber":
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss: {loss}")

    def get_metrics(self, num_outputs: int) -> MetricCollection:
        return MetricCollection(
            {
                "mae": MeanAbsoluteError(num_outputs=num_outputs),
                "mse": MeanSquaredError(num_outputs=num_outputs, squared=True),
                "rmse": MeanSquaredError(num_outputs=num_outputs, squared=False),
                # "r2": R2Score(num_outputs=num_outputs),  # does not work with single batch
                "spearman": SpearmanCorrCoef(num_outputs=num_outputs),  # does not work with single batch
                "pearson": PearsonCorrCoef(num_outputs=num_outputs),  # does not work with single batch
            }
        )

    def reduce_log_reset(self, metrics: MetricCollection):
        scores = metrics.compute()
        metrics.reset()

        means = {f'{k}_mean': v.mean() for k, v in scores.items()}
        stds = {f'{k}_std': v.std() for k, v in scores.items()}
        self.log_dict(means)
        self.log_dict(stds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        z = pool(z, strategy=self.pooling)
        y_hat = self.head(z)  # [B, 1]
        return y_hat

    def compute_loss(self, batch, y_hat):
        y = glom(batch, self.target_key)
        y = y.unsqueeze(1) if y.ndim == 1 else y  # [B, 1]
        y = y.float()
        loss = self.criterion(y_hat, y)
        return y, loss
    
    def shared_step(self, batch: dict, batch_idx: int):
        x = glom(batch, self.batch_key) if self.batch_key is not None else batch

        z = self.backbone(x)
        z = pool(z, strategy=self.pooling)

        y_hat = self.head(z)
        assert y_hat.ndim == 2
        # assert y_hat.shape[1] == 1, f"Expected y_hat [B,1], got {tuple(y_hat.shape)}"
        
        return z, y_hat

    def training_step(self, batch, batch_idx: int):
        z, y_hat = self.shared_step(batch, batch_idx)
        y, loss = self.compute_loss(batch, y_hat)
        batch_size = int(y.shape[0])

        self.log("loss/train", loss, on_step=True, on_epoch=True, batch_size=batch_size)

        # metrics
        self.train_metrics.update(y_hat, y)

        batch["loss"] = loss
        batch["y_hat"] = y_hat.detach().cpu()
        batch["y"] = y.detach().cpu()
        batch["z"] = z.detach().cpu()
        return batch

    def on_train_epoch_end(self) -> None:
        if not self.trainer.fast_dev_run:
            # NOTE: we log here to support metrics that are only defined for B > 1 (e.g., R2Score)
            metrics = self.train_metrics
            if self.num_outputs == 1:
                self.log_dict(metrics)
            else:
                self.reduce_log_reset(metrics)


    def validation_step(self, batch, batch_idx: int):
        z, y_hat = self.shared_step(batch, batch_idx)
        y, loss = self.compute_loss(batch, y_hat)
        batch_size = int(y.shape[0])

        self.log("loss/val", loss, on_step=False, on_epoch=True, batch_size=batch_size)

        self.valid_metrics.update(y_hat, y)

        batch["loss"] = loss
        batch["y_hat"] = y_hat.detach().cpu()
        batch["y"] = y.detach().cpu()
        batch["z"] = z.detach().cpu()
        return batch

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.fast_dev_run:
            metrics = self.valid_metrics
            if self.num_outputs == 1:
                self.log_dict(metrics)
            else:
                self.reduce_log_reset(metrics)

    def test_step(self, batch, batch_idx: int):
        z, y_hat = self.shared_step(batch, batch_idx)
        y, loss = self.compute_loss(batch, y_hat)
        batch_size = int(y.shape[0])

        self.log("loss/test", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.test_metrics.update(y_hat, y)

        batch["loss"] = loss
        batch["y_hat"] = y_hat.detach().cpu()
        batch["y"] = y.detach().cpu()
        batch["z"] = z.detach().cpu()
        return batch

    def on_test_epoch_end(self) -> None:
        if not self.trainer.fast_dev_run:
            metrics = self.test_metrics
            if self.num_outputs == 1:
                self.log_dict(metrics)
            else:
                self.reduce_log_reset(metrics)

    def predict_step(self, batch, batch_idx: int):
        z, y_hat = self.shared_step(batch, batch_idx)

        batch["y_hat"] = y_hat.detach().cpu()
        batch["z"] = z.detach().cpu()

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
            assert max_epochs is not None, "trainer.max_epochs is None"
        except AttributeError or AssertionError as e:
            logger.warning(f'`max_epoch not found in trainer ({e}), using module max_epochs={self.max_epochs}`')
            max_epochs = self.max_epochs

        num_warmup_epochs = self.num_warmup_epochs
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-2,
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


class RegressionLitMultiData(RegressionLit):
    def test_step(self, batch, batch_idx: int, dataloader_idx=0):
        z, y_hat = self.shared_step(batch, batch_idx)
        y, loss = self.compute_loss(batch, y_hat)
        batch_size = int(y.shape[0])

        self.log(
            f"loss/test",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.test_metrics_list[dataloader_idx].update(y_hat, y)

        batch["loss"] = loss
        batch["y_hat"] = y_hat.detach().cpu()
        batch["y"] = y.detach().cpu()
        batch["z"] = z.detach().cpu()
        return batch

    def on_test_start(self):
        num_loaders = len(self.trainer.test_dataloaders)

        self.test_metrics_list = nn.ModuleList(
            [
                self.test_metrics.clone(prefix=f"test_loader_{i}/")
                for i in range(num_loaders)
            ]
        )

    def reduce_log_reset(self, metrics: MetricCollection, prefix: str = ""):
        scores = metrics.compute()
        metrics.reset()

        means = {f"{prefix}{k}_mean": v.mean() for k, v in scores.items()}
        stds = {f"{prefix}{k}_std": v.std() for k, v in scores.items()}

        self.log_dict(means)
        self.log_dict(stds)

    def on_test_epoch_end(self) -> None:
        if self.trainer.fast_dev_run:
            return

        # Loop over each dataloader's metrics
        for idx, metrics in enumerate(self.test_metrics_list):
            prefix = f"test_loader_{idx}/"

            if self.num_outputs == 1:
                # Single-output case: just log metrics
                self.log_dict({f"{prefix}{k}": v for k, v in metrics.compute().items()})
                metrics.reset()
            else:
                # Multi-output: use your reduce_log_reset method with prefix
                self.reduce_log_reset(metrics, prefix=prefix)
