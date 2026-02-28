from collections import Counter

import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from glom import glom
from torchmetrics.classification import ConfusionMatrix

from ai4bmr_learn.metrics.classification import get_metric_collection


class MILLit(L.LightningModule):

    def __init__(self,
                 backbone: nn.Module,
                 num_classes: int,
                 head: nn.Module | None = None,
                 lr_head: float = 1e-3,
                 lr_backbone: float = 1e-4,
                 weight_decay: float = 0.01,
                 batch_key: str | None = 'image',
                 target_key: str = 'label',
                 weight: torch.Tensor | None = None,
                 log_confusion_matrix_every_n_epochs: int = 25,
                 schedule: str | None = None,
                 eta: float = 0.0,
                 max_epochs: int = 1000,
                 num_warmup_epochs: int = 10
                 ):
        super().__init__()

        self.save_hyperparameters(ignore=["backbone", "head"])

        self.backbone = backbone
        self.head = head or nn.Linear(backbone.output_dim, num_classes)

        self.batch_key = batch_key
        self.target_key = target_key

        self.lr_head = lr_head
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

        self.criterion = nn.CrossEntropyLoss(weight=weight)

        # METRICS
        self.num_classes = num_classes
        task = "binary" if num_classes == 2 else "multiclass"

        metrics = get_metric_collection(num_classes=num_classes)
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        normalize = None
        self.cm_train = ConfusionMatrix(task=task, normalize=normalize, num_classes=num_classes)
        self.cm_val = ConfusionMatrix(task=task, normalize=normalize, num_classes=num_classes)
        self.cm_test = ConfusionMatrix(task=task, normalize=normalize, num_classes=num_classes)
        self.log_confusion_matrix_every_n_epochs = log_confusion_matrix_every_n_epochs

        # STATS
        self.train_stats = {'train/num_samples': 0, 'class_cnt': Counter()}
        self.val_stats = {'val/num_samples': 0, 'class_cnt': Counter()}

        # OPTIM
        self.eta = eta
        self.max_epochs = max_epochs
        self.num_warmup_epochs = num_warmup_epochs
        assert schedule in [None, 'cosine'], "Only 'cosine' schedule is supported currently."
        self.schedule = schedule

    def shared_step(self, batch, batch_idx):
        data = glom(batch, self.batch_key) if self.batch_key is not None else batch

        z, attn, attn_logits = self.backbone(data, return_attn_and_logits=True)
        logits = self.head(z)
        targets = glom(batch, self.target_key).long()

        loss = self.criterion(logits, targets)
        assert not torch.isnan(loss), f"Loss is NaN for {batch['sample_id']}"

        return z, logits, targets, loss, attn, attn_logits

    def on_train_epoch_start(self):
        self.train_stats['train/num_samples'] = 0
        self.train_stats['class_cnt'] = Counter()

    def on_train_epoch_end(self):
        class_cnt = self.train_stats.pop('class_cnt')
        class_counts = {f'train/class_{k}': v for k, v in class_cnt.items()}
        self.train_stats.update(class_counts)

        self.log_dict(self.train_stats)

        if self.current_epoch % self.log_confusion_matrix_every_n_epochs == 0:
            fig, ax = self.cm_train.plot()
            self.logger.experiment.log({'train/confusion_matrix': fig})
            plt.close('all')
        self.cm_train.reset()

        self.train_stats['train/num_samples'] = 0
        self.train_stats['class_cnt'] = Counter()

    def training_step(self, batch, batch_idx):
        z, logits, targets, loss, attn, attn_logits = self.shared_step(batch, batch_idx)  # added attn
        batch_size = targets.shape[0]

        # metrics
        preds = torch.argmax(logits, dim=1).long()
        self.train_metrics(logits, targets)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, batch_size=batch_size)

        self.cm_train(preds, targets)

        # loss
        self.log("loss/train", loss, on_step=True, on_epoch=True, batch_size=batch_size)

        # stats
        self.train_stats['train/num_samples'] += batch_size
        self.train_stats['class_cnt'].update(targets.tolist())

        batch['loss'] = loss
        batch['z'] = z
        batch['attn'] = attn
        batch['attn_logits'] = attn_logits
        return batch

    def on_validation_epoch_start(self):
        self.val_stats['val/num_samples'] = 0
        self.val_stats['class_cnt'] = Counter()
        self.validation_step_outputs = {}  # added to store attention weights

    def on_validation_epoch_end(self):
        class_cnt = self.val_stats.pop('class_cnt')
        class_counts = {f'val/class_{k}': v for k, v in class_cnt.items()}
        self.val_stats.update(class_counts)

        self.log_dict(self.val_stats)

        if self.current_epoch % self.log_confusion_matrix_every_n_epochs == 0:
            fig, ax = self.cm_val.plot()
            self.logger.experiment.log({'val/confusion_matrix': fig})
            plt.close('all')
        self.cm_val.reset()

        self.val_stats['val/num_samples'] = 0
        self.val_stats['class_cnt'] = Counter()

    def validation_step(self, batch, batch_idx):
        z, logits, targets, loss, attn, attn_logits = self.shared_step(batch, batch_idx)  # added attn
        batch_size = targets.shape[0]

        # metrics
        preds = torch.argmax(logits, dim=1).long()
        self.valid_metrics(logits, targets)
        self.log_dict(self.valid_metrics, on_step=False, on_epoch=True, batch_size=batch_size)

        self.cm_val(preds, targets)

        # loss
        self.log("loss/val", loss, batch_size=batch_size)

        # stats
        self.val_stats['val/num_samples'] += batch_size
        self.val_stats['class_cnt'].update(targets.tolist())

        batch['loss'] = loss
        batch['z'] = z
        batch['attn'] = attn
        batch['attn_logits'] = attn_logits

        return batch

    def test_step(self, batch, batch_idx):
        z, logits, targets, loss, attn, attn_logits = self.shared_step(batch, batch_idx)  # added attn
        batch_size = targets.shape[0]

        # metrics
        preds = torch.argmax(logits, dim=1).long()
        self.test_metrics(logits, targets)
        self.log_dict(self.test_metrics, batch_size=batch_size)

        self.cm_test(preds, targets)

        # loss
        self.log("loss/test", loss, batch_size=batch_size)

        batch['loss'] = loss
        batch['z'] = z
        batch['attn'] = attn
        batch['attn_logits'] = attn_logits
        return batch

    def on_test_end(self) -> None:
        fig, ax = self.cm_test.plot()
        self.cm_test.reset()
        self.logger.experiment.log({'test/confusion_matrix': fig})
        plt.close('all')

    def configure_optimizers(self):
        optimizer = optim.AdamW([
            {'params': self.head.parameters(), 'lr': self.lr_head},
            {'params': filter(lambda p: p.requires_grad, self.backbone.parameters()), 'lr': self.lr_backbone}
        ], weight_decay=self.weight_decay)

        if self.schedule is None:
            return optimizer

        max_epochs = getattr(self.trainer, "max_epochs", None) or self.max_epochs

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
