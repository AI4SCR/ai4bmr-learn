from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from glom import glom
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import AUROC, Accuracy, F1Score, Precision, Recall
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

from ai4bmr_learn.models.mil import AggregationOutput


def detach_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: detach_value(val) for key, val in value.items()}
    return value


class BaseMILLit(L.LightningModule, ABC):
    def __init__(
        self,
        aggregator: nn.Module,
        head: nn.Module,
        bag_key: str = "bag",
        mask_key: str = "mask",
        lr_head: float = 1e-3,
        lr_aggregator: float = 1e-4,
        weight_decay: float = 0.01,
        schedule: str | None = None,
        eta: float = 0.0,
        max_epochs: int = 1000,
        num_warmup_epochs: int = 10,
    ):
        super().__init__()
        assert schedule in [None, "cosine"], "unknown schedule"

        self.aggregator = aggregator
        self.head = head
        self.bag_key = bag_key
        self.mask_key = mask_key

        self.lr_head = lr_head
        self.lr_aggregator = lr_aggregator
        self.weight_decay = weight_decay
        self.schedule = schedule
        self.eta = eta
        self.max_epochs = max_epochs
        self.num_warmup_epochs = num_warmup_epochs

    def forward(self, bag: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, AggregationOutput]:
        aggregation = self.aggregator(bag, mask)
        assert isinstance(aggregation, AggregationOutput), "aggregator must return AggregationOutput"
        output = self.head(aggregation.embedding)
        return output, aggregation

    def get_bag_and_mask(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        bag = glom(batch, self.bag_key)
        mask = glom(batch, self.mask_key)
        assert isinstance(bag, torch.Tensor), "bag must be a tensor"
        assert isinstance(mask, torch.Tensor), "mask must be a tensor"
        return bag, mask

    def shared_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        bag, mask = self.get_bag_and_mask(batch)
        output, aggregation = self.forward(bag=bag, mask=mask)
        target = self.get_target(batch)
        loss = self.compute_loss(output=output, target=target, batch=batch)
        return {
            "loss": loss,
            "output": output,
            "target": target,
            "aggregation": aggregation,
        }

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        result = self.shared_step(batch=batch, batch_idx=batch_idx)
        return self.log_and_format_step(stage="train", result=result, batch=batch)

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        result = self.shared_step(batch=batch, batch_idx=batch_idx)
        return self.log_and_format_step(stage="val", result=result, batch=batch)

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        result = self.shared_step(batch=batch, batch_idx=batch_idx)
        return self.log_and_format_step(stage="test", result=result, batch=batch)

    def predict_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        bag, mask = self.get_bag_and_mask(batch)
        output, aggregation = self.forward(bag=bag, mask=mask)
        return self.format_output(result={"output": output, "aggregation": aggregation})

    def log_and_format_step(self, stage: str, result: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
        output = result["output"]
        batch_size = int(output.shape[0])

        self.log(f"loss/{stage}", result["loss"], on_step=stage == "train", on_epoch=True, batch_size=batch_size)

        metrics = getattr(self, f"{stage}_metrics", None)
        assert isinstance(metrics, MetricCollection), f"missing {stage}_metrics"
        self.update_metrics(metrics=metrics, output=output, target=result["target"], batch=batch)

        return self.format_output(result=result)

    def on_train_epoch_end(self) -> None:
        self.log_epoch_metrics(stage="train")

    def on_validation_epoch_end(self) -> None:
        self.log_epoch_metrics(stage="val")

    def on_test_epoch_end(self) -> None:
        self.log_epoch_metrics(stage="test")

    def format_output(self, result: dict[str, Any]) -> dict[str, Any]:
        output = result["output"]
        aggregation = result["aggregation"]
        assert isinstance(output, torch.Tensor), "output must be a tensor"
        assert isinstance(aggregation, AggregationOutput), "aggregation must be AggregationOutput"

        formatted = {
            "output": output.detach().cpu(),
            "prediction": self.format_prediction(output).detach().cpu(),
            "embedding": aggregation.embedding.detach().cpu(),
        }
        if aggregation.weights is not None:
            formatted["weights"] = aggregation.weights.detach().cpu()
        if aggregation.logits is not None:
            formatted["logits"] = aggregation.logits.detach().cpu()
        if "loss" in result:
            formatted["loss"] = result["loss"]
        if "target" in result:
            formatted["target"] = detach_value(result["target"])
        return formatted

    def configure_optimizers(self):
        param_groups = [
            *self.get_parameter_groups(module=self.aggregator, lr=self.lr_aggregator),
            *self.get_parameter_groups(module=self.head, lr=self.lr_head),
        ]
        assert param_groups, "no trainable parameters"

        optimizer = optim.AdamW(param_groups)
        if self.schedule is None:
            return optimizer

        max_epochs = self.get_max_epochs()
        assert max_epochs > self.num_warmup_epochs, "warmup exceeds max_epochs"
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-2,
            end_factor=1.0,
            total_iters=self.num_warmup_epochs,
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - self.num_warmup_epochs,
            eta_min=self.eta,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.num_warmup_epochs],
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def get_max_epochs(self) -> int:
        try:
            trainer = self.trainer
        except RuntimeError:
            trainer = None
        max_epochs = getattr(trainer, "max_epochs", None) or self.max_epochs
        return int(max_epochs)

    def get_parameter_groups(self, module: nn.Module, lr: float) -> list[dict[str, Any]]:
        decay = []
        no_decay = []
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)

        groups = []
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": self.weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
        return groups

    def log_epoch_metrics(self, stage: str) -> None:
        metrics = getattr(self, f"{stage}_metrics", None)
        assert isinstance(metrics, MetricCollection), f"missing {stage}_metrics"
        self.log_dict(metrics)

    @abstractmethod
    def get_target(self, batch: dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, output: torch.Tensor, target: Any, batch: dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def update_metrics(
        self,
        metrics: MetricCollection,
        output: torch.Tensor,
        target: Any,
        batch: dict[str, Any],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def format_prediction(self, output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ClassificationMILLit(BaseMILLit):
    def __init__(
        self,
        aggregator: nn.Module,
        head: nn.Module,
        num_classes: int,
        target_key: str = "target",
        class_weight: torch.Tensor | None = None,
        metric_names: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(aggregator=aggregator, head=head, **kwargs)
        assert num_classes > 1, "num_classes must exceed 1"
        self.num_classes = num_classes
        self.target_key = target_key
        self.criterion = nn.CrossEntropyLoss(weight=class_weight)

        metrics = self.get_metrics(num_classes=num_classes, metric_names=metric_names)
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.save_hyperparameters(ignore=["aggregator", "head", "class_weight"])

    @staticmethod
    def get_metrics(num_classes: int, metric_names: list[str] | None = None) -> MetricCollection:
        if metric_names is None:
            return MetricCollection(
                {
                    "accuracy-micro": Accuracy(task="multiclass", average="micro", num_classes=num_classes),
                    "accuracy-macro": Accuracy(task="multiclass", average="macro", num_classes=num_classes),
                    "recall-macro": Recall(task="multiclass", average="macro", num_classes=num_classes),
                    "precision-macro": Precision(task="multiclass", average="macro", num_classes=num_classes),
                    "f1-macro": F1Score(task="multiclass", average="macro", num_classes=num_classes),
                }
            )

        metrics = {}
        for name in metric_names:
            match name:
                case "balanced_accuracy":
                    metrics[name] = Recall(task="multiclass", average="macro", num_classes=num_classes)
                case "accuracy":
                    metrics[name] = Accuracy(task="multiclass", average="micro", num_classes=num_classes)
                case "accuracy_macro":
                    metrics[name] = Accuracy(task="multiclass", average="macro", num_classes=num_classes)
                case "precision_macro":
                    metrics[name] = Precision(task="multiclass", average="macro", num_classes=num_classes)
                case "recall_macro":
                    metrics[name] = Recall(task="multiclass", average="macro", num_classes=num_classes)
                case "f1" | "f1_macro":
                    metrics[name] = F1Score(task="multiclass", average="macro", num_classes=num_classes)
                case "roc_auc":
                    metrics[name] = AUROC(task="multiclass", num_classes=num_classes)
                case _:
                    raise ValueError(f"Unknown metric: {name}")
        return MetricCollection(metrics)

    def get_target(self, batch: dict[str, Any]) -> torch.Tensor:
        target = glom(batch, self.target_key)
        assert isinstance(target, torch.Tensor), "target must be a tensor"
        return target.long().reshape(-1)

    def compute_loss(self, output: torch.Tensor, target: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        assert output.ndim == 2, f"Expected logits [B,C], got {tuple(output.shape)}"
        assert output.shape[1] == self.num_classes, f"Expected C={self.num_classes}, got {output.shape[1]}"
        assert output.shape[0] == target.shape[0], "batch size mismatch"
        return self.criterion(output, target)

    def update_metrics(
        self,
        metrics: MetricCollection,
        output: torch.Tensor,
        target: torch.Tensor,
        batch: dict[str, Any],
    ) -> None:
        metrics.update(output, target)

    def format_prediction(self, output: torch.Tensor) -> torch.Tensor:
        return output.argmax(dim=1)


class RegressionMILLit(BaseMILLit):
    def __init__(
        self,
        aggregator: nn.Module,
        head: nn.Module,
        target_key: str = "target",
        num_outputs: int = 1,
        loss: str = "mse",
        metric_names: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(aggregator=aggregator, head=head, **kwargs)
        assert num_outputs > 0, "num_outputs must be positive"
        self.target_key = target_key
        self.num_outputs = num_outputs
        self.criterion = self.get_loss(loss=loss)

        metrics = self.get_metrics(num_outputs=num_outputs, metric_names=metric_names)
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.save_hyperparameters(ignore=["aggregator", "head"])

    @staticmethod
    def get_loss(loss: str) -> nn.Module:
        match loss:
            case "mse":
                return nn.MSELoss()
            case "huber":
                return nn.SmoothL1Loss()
            case _:
                raise ValueError(f"Unknown loss: {loss}")

    @staticmethod
    def get_metrics(num_outputs: int, metric_names: list[str] | None = None) -> MetricCollection:
        if metric_names is None:
            return MetricCollection(
                {
                    "mae": MeanAbsoluteError(num_outputs=num_outputs),
                    "mse": MeanSquaredError(num_outputs=num_outputs, squared=True),
                    "rmse": MeanSquaredError(num_outputs=num_outputs, squared=False),
                }
            )

        metrics = {}
        for name in metric_names:
            match name:
                case "mse" | "mean_squared_error":
                    metrics[name] = MeanSquaredError(num_outputs=num_outputs, squared=True)
                case "rmse":
                    metrics[name] = MeanSquaredError(num_outputs=num_outputs, squared=False)
                case "mae" | "mean_absolute_error":
                    metrics[name] = MeanAbsoluteError(num_outputs=num_outputs)
                case "r2":
                    metrics[name] = R2Score()
                case _:
                    raise ValueError(f"Unknown metric: {name}")
        return MetricCollection(metrics)

    def get_target(self, batch: dict[str, Any]) -> torch.Tensor:
        target = glom(batch, self.target_key)
        assert isinstance(target, torch.Tensor), "target must be a tensor"
        target = target.float()
        return target.unsqueeze(1) if target.ndim == 1 else target

    def compute_loss(self, output: torch.Tensor, target: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        output = output.unsqueeze(1) if output.ndim == 1 else output
        assert output.shape == target.shape, f"Expected output {tuple(target.shape)}, got {tuple(output.shape)}"
        return self.criterion(output, target)

    def update_metrics(
        self,
        metrics: MetricCollection,
        output: torch.Tensor,
        target: torch.Tensor,
        batch: dict[str, Any],
    ) -> None:
        output = output.unsqueeze(1) if output.ndim == 1 else output
        metrics.update(output, target)

    def format_prediction(self, output: torch.Tensor) -> torch.Tensor:
        return output


class TorchSurvMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("risk", default=[], dist_reduce_fx="cat")
        self.add_state("time", default=[], dist_reduce_fx="cat")
        self.add_state("event", default=[], dist_reduce_fx="cat")

    def update(self, risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> None:
        self.risk.append(risk.detach().reshape(-1))
        self.time.append(time.detach().reshape(-1).float())
        self.event.append(event.detach().reshape(-1).bool())

    def _get_variables(self):
        if not self.risk:
            return None

        risk = torch.cat(self.risk)
        time = torch.cat(self.time)
        event = torch.cat(self.event)
        if risk.numel() < 2 or event.sum() == 0:
            return None
        return risk, event, time

    def compute(self) -> torch.Tensor:
        raise NotImplementedError()


class ConcordanceIndexMetric(TorchSurvMetric):
    def __init__(self):
        super().__init__()
        self.default_metric_value = 0.0

    def _get_cindex(self, variables):
        risk, event, time = variables

        from torchsurv.metrics.cindex import ConcordanceIndex

        cindex = ConcordanceIndex()
        return cindex, cindex(risk, event, time)

    def compute(self) -> torch.Tensor:
        variables = self._get_variables()
        if variables is None:
            return torch.tensor(self.default_metric_value)
        _, value = self._get_cindex(variables)
        return torch.as_tensor(value).to(variables[0]).squeeze()


class ConcordanceIndexPValueMetric(ConcordanceIndexMetric):
    def __init__(self):
        super().__init__()
        self.default_metric_value = 1.0

    def compute(self) -> torch.Tensor:
        variables = self._get_variables()
        if variables is None:
            return torch.tensor(self.default_metric_value)

        cindex, _ = self._get_cindex(variables)

        pvalue = cindex.p_value(alternative="greater")
        return torch.as_tensor(pvalue).to(variables[0]).squeeze()


class InegratedBrierScoreMetric(TorchSurvMetric):
    def __init__(self):
        raise NotImplementedError()
        super().__init__()
        self.default_metric_value = torch.nan

    def compute(self) -> torch.Tensor:
        raise NotImplementedError()

        variables = self._get_variables()
        if variables is None:
            return torch.tensor(self.default_metric_value)
        risk, event, time = variables

        from torchsurv.loss import cox
        from torchsurv.metrics.brier_score import BrierScore

        # FIXME baseline surv should be estimated on the (whole) train set!
        # - Given the current model, pass whole train set through momdel
        # - Estimate baseline_survival_function on that data
        # - Then calculate survival_function_cox on train/val/test data (risk_eval, time_eval)
        # - See https://opensource.nibr.com/torchsurv/_autosummary/torchsurv.loss.cox.html
        # - https://opensource.nibr.com/torchsurv/notebooks/introduction.html, cell 14, 15 & 20

        event_train, time_train = ...  # get whole train set
        risk_train = ...  # pass through model to get hazard on train set
        baseline_surv = cox.baseline_survival_function(risk_train, event_train, time_train)

        surv = cox.survival_function_cox(baseline_surv, risk, time)

        brier_score = BrierScore()
        _ = brier_score(surv, event, time)
        value = brier_score.integral()
        return torch.as_tensor(value, dtype=torch.float32, device=risk.device).squeeze()


class SurvivalMILLit(BaseMILLit):
    def __init__(
        self,
        aggregator: nn.Module,
        head: nn.Module,
        time_key: str = "time",
        event_key: str = "event",
        metric_names: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(aggregator=aggregator, head=head, **kwargs)
        self.time_key = time_key
        self.event_key = event_key

        metrics = self.get_metrics(metric_names)
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.train_total_batches = 0
        self.train_zero_event_batches = 0

        self.save_hyperparameters(ignore=["aggregator", "head"])

    def on_train_epoch_start(self) -> None:
        self.train_total_batches = 0
        self.train_zero_event_batches = 0

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        self.train_total_batches += 1

        target = self.get_target(batch)
        event = target["event"]

        if not event.any():
            self.train_zero_event_batches += 1
            return None

        return super().training_step(
            batch=batch,
            batch_idx=batch_idx,
        )

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()

        zero_frac = self.train_zero_event_batches / max(self.train_total_batches, 1)

        self.log(
            "train/zero_event_batches",
            self.train_zero_event_batches,
            prog_bar=False,
            sync_dist=True,
        )

        self.log(
            "train/zero_event_fraction",
            zero_frac,
            prog_bar=True,
            sync_dist=True,
        )

    @staticmethod
    def get_metrics(metric_names: list[str] | None = None) -> MetricCollection:
        if metric_names is None:
            return MetricCollection(
                {
                    "cindex": ConcordanceIndexMetric(),
                    "cindex_pvalue": ConcordanceIndexPValueMetric(),
                }
            )

        metrics = {}
        for name in metric_names:
            match name:
                case "cindex" | "concordance_index_metric" | "c_index":
                    metrics[name] = ConcordanceIndexMetric()
                case "cindex_pvalue" | "c_index_p_value":
                    metrics[name] = ConcordanceIndexPValueMetric()
                case "brier" | "brier_score" | "integrated_brier_score":
                    metrics[name] = InegratedBrierScoreMetric()
                case _:
                    raise ValueError(f"Unknown metric: {name}")
        return MetricCollection(metrics)

    def get_target(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        time = glom(batch, self.time_key)
        event = glom(batch, self.event_key)
        assert isinstance(time, torch.Tensor), "time must be a tensor"
        assert isinstance(event, torch.Tensor), "event must be a tensor"
        time = time.float().reshape(-1)
        event = event.bool().reshape(-1)
        assert time.shape == event.shape, "time and event shapes differ"
        return {"time": time, "event": event}

    def compute_loss(
        self,
        output: torch.Tensor,
        target: dict[str, torch.Tensor],
        batch: dict[str, Any],
    ) -> torch.Tensor:
        from torchsurv.loss import cox

        risk = output.reshape(-1)
        time = target["time"]
        event = target["event"]
        assert risk.shape == time.shape, "risk and time shapes differ"
        assert (
            event.any()
        ), "no events in batch. This should not happen in training as this is handled in `training_step`. At validation or test time this should be avoided by feeding the full val or test set"
        return cox.neg_partial_log_likelihood(risk, event, time)

    def update_metrics(
        self,
        metrics: MetricCollection,
        output: torch.Tensor,
        target: dict[str, torch.Tensor],
        batch: dict[str, Any],
    ) -> None:
        metrics.update(output.reshape(-1), target["time"], target["event"])

    def format_prediction(self, output: torch.Tensor) -> torch.Tensor:
        return output.reshape(-1)
