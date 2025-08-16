import lightning as L
import torch
from glom import glom
from ai4bmr_learn.metrics.classification import get_metric_collection
from ai4bmr_learn.callbacks.cache import TrainCache
from loguru import logger

class LogisticRegression(L.LightningModule):

    def __init__(self, target_key: str, batch_key: str, num_classes: int | None, **kwargs):
        super().__init__()
        from sklearn.linear_model import LogisticRegression
        self.automatic_optimization = False  # needed if you do not want to return a loss in `training_step`

        # data keys
        self.batch_key = batch_key
        self.target_key = target_key

        # model
        self.model = LogisticRegression(**kwargs)

        # metrics
        metrics = get_metric_collection(num_classes=num_classes)
        self.train_metrics = metrics.clone(prefix="logistic_reg/train/")
        self.val_metrics   = metrics.clone(prefix="logistic_reg/val/")
        self.test_metrics  = metrics.clone(prefix="logistic_reg/test/")

    def get_data_and_targets(self, batch, return_targets: bool = True):

        if isinstance(batch, dict):
            data = glom(batch, self.batch_key)
            targets = glom(batch, self.target_key) if return_targets else None

        elif isinstance(batch, list):
            data = [glom(b, self.batch_key) for b in batch]
            targets = [glom(b, self.target_key) for b in batch] if return_targets else None
            data = torch.concat(data)
            targets = torch.concat(targets) if return_targets else None
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")

        return data, targets

    def training_step(self, batch, batch_idx):
        # NOTE: here we could do online-learning with `partial_fit` if supported.
        return batch

    def on_validation_start(self) -> None:
        # NOTE: hooke order ➡️ https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks

        # find active TrainCache callback
        train_cache = next(cb for cb in self.trainer.callbacks if isinstance(cb, TrainCache))
        outputs = train_cache.outputs

        x, y = [],  []
        for batch in outputs:
            x_, y_ = self.get_data_and_targets(batch)
            x.append(x_.cpu())
            y.append(y_.cpu())

        if self.trainer.sanity_checking:
            return

        # collect samples
        x = torch.cat(x).numpy()
        y = torch.cat(y).numpy()

        # fit
        logger.info('Fitting model...')
        self.model.fit(X=x, y=y)

        # metrics
        preds = self.model.predict(X=x)
        preds = torch.tensor(preds, device=self.device)
        targets = torch.tensor(y, device=self.device)
        self.train_metrics(preds, targets)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):

        data, targets = self.get_data_and_targets(batch)

        if self.trainer.sanity_checking:
            return

        preds = self.model.predict(data.cpu().numpy())
        preds = torch.tensor(preds, device=self.device)
        self.val_metrics(preds, targets)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        data, targets = self.get_data_and_targets(batch)

        if self.trainer.sanity_checking:
            return

        preds = self.model.predict(data.cpu().numpy())
        preds = torch.tensor(preds, device=self.device)
        self.test_metrics(preds, targets)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

    def prediction_step(self, batch, batch_idx):
        data, _ = self.get_data_and_targets(batch, return_targets=False)
        preds = self.model.predict(data.cpu().numpy())
        return torch.tensor(preds, device=self.device)

    def configure_optimizers(self):
        return None