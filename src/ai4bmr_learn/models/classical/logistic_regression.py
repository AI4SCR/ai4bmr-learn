import lightning as L
from glom import glom
from ai4bmr_learn.metrics.classification import get_metric_collection
from loguru import logger
import torch

class LogisticRegression(L.LightningModule):

    def __init__(self, target_key: str, batch_key: str, num_classes: int | None, **kwargs):
        from sklearn.linear_model import LogisticRegression

        super().__init__()

        # data
        self.batch_key = batch_key
        self.target_key = target_key

        # model
        self.model = LogisticRegression()

        # metrics
        self.num_classes = num_classes
        metrics = get_metric_collection(num_classes=num_classes)
        self.train_metrics = metrics.clone(prefix="logistic_regression/train/")
        self.val_metrics = metrics.clone(prefix="logistic_regression/val/")
        self.test_metrics = metrics.clone(prefix="logistic_regression/test/")

    def get_data_and_targets(self, batch, return_targets: bool = True):
        if isinstance(batch, dict):
            data = glom(batch, self.batch_key).squeeze(0)
            targets = glom(batch, self.target_key).squeeze(0) if return_targets else None

        elif isinstance(batch, list):
            data = [glom(batch, self.batch_key).squeeze(0) for batch in batch]
            targets = [glom(batch, self.target_key).squeeze(0) for batch in batch]  if return_targets else None

            data = torch.concat(data)
            targets = torch.concat(targets)  if return_targets else None

        else:
            raise ValueError(f'Unsupported batch type: {type(batch)}')
        return data, targets

    def training_step(self, batch, batch_idx):
        data, targets = self.get_data_and_targets(batch)

        if self.trainer.sanity_checking:
            return

        logger.info(f'Fitting model...')
        self.model.fit(X=data.cpu().numpy(), y=targets.cpu().numpy())

        preds = self.model.predict(X=data.cpu().numpy())
        preds = torch.tensor(preds, device=data.device)
        self.train_metrics(preds, targets)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        data, targets = self.get_data_and_targets(batch)

        if self.trainer.sanity_checking:
            return

        preds = self.model.predict(X=data.cpu().numpy())
        preds = torch.tensor(preds, device=data.device)
        self.val_metrics(preds, targets)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        data, targets = self.get_data_and_targets(batch)

        preds = self.model.predict(X=data.cpu().numpy())
        preds = torch.tensor(preds, device=data.device)
        self.test_metrics(preds, targets)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

    def prediction_step(self, batch, batch_idx):
        data, _ = self.get_data_and_targets(batch, return_targets=False)

        preds = self.model.predict(X=data.cpu().numpy())
        preds = torch.tensor(preds, device=data.device)
        return preds

    def configure_optimizers(self):
        pass