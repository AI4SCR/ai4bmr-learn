import lightning as L
import torch.optim as optim
import torch.nn as nn
from ai4bmr_learn.metrics.classification import get_metric_collection
from glom import glom
import torch
import numpy as np
from ai4bmr_learn.utils.pooling import pool
import torch.nn as nn
import lightning as L
import torch.optim as optim
from ai4bmr_learn.metrics.classification import get_metric_collection
from torchmetrics.classification import ConfusionMatrix
from glom import glom
from collections import Counter
from ai4bmr_learn.utils.pooling import pool
import torch
from torch import nn

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def freeze(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    return model


class MILTrainerExtended(L.LightningModule):
    def __init__(self,
                 backbone: nn.Module,
                 mil: nn.Module,
                 head: nn.Module,
                 num_classes: int,
                 lr_head: float = 1e-3,
                 lr_backbone: float = 1e-4,
                 lr_mil: float = 1e-4,
                 weight_decay: float = 0.01,
                 freeze_backbone: bool = False,
                 pooling: str | None = None,
                 batch_key: str | None = 'image',
                 data_keys: list[str] | None = None,
                 target_key: str = 'target',
                 attention_key: str = 'attention',
                 as_kwargs: bool = False,
                 weight: torch.Tensor | None = None
                 ):

        super().__init__()

        # MODULES
        self.freeze_backbone = freeze_backbone
        self.backbone = backbone
        if self.freeze_backbone:
            freeze(self.backbone)
        self.mil = mil
        self.head = head

        # LOSS
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

        # STATS
        self.train_stats = {'train/num_samples': 0, 'class_cnt': Counter()}
        self.val_stats = {'val/num_samples': 0, 'class_cnt': Counter()}

        # PARAMS
        self.lr_head = lr_head
        self.lr_mil = lr_mil
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.freeze_backbone = freeze_backbone
        self.pooling = pooling

        # DATA
        self.batch_key = batch_key
        self.data_keys = data_keys
        if self.data_keys is None:
            self.data_keys = [batch_key]
        self.target_key = target_key
        self.attention_key = attention_key
        self.as_kwargs = as_kwargs

        self.save_hyperparameters(ignore=["backbone", "mil", "head"])

    def shared_step(self, batch, batch_idx: int | None = None):
        bags = glom(batch, self.batch_key)
        assert isinstance(bags, torch.Tensor)

        targets = glom(batch, self.target_key)
        assert targets.ndim == 1  # [B,]
        assert len(targets) == len(bags)

        B, M, *D = bags.shape

        zs = []
        # NOTE: we need to feed the bags separately to the backbone, since they are designed for [B, D] inputs
        for i, bag in enumerate(bags):

            z = self.backbone(**bag) if self.as_kwargs else self.backbone(bag)
            z = pool(z, strategy=self.pooling)
            zs.append(z)

            del z

        zs = torch.stack(zs)  # [B, M, Z]
        assert zs.shape[0] == B and zs.shape[1] == M

        z, attn, attn_logits = self.mil(zs)
        logits = self.head(z)
        loss = self.criterion(logits, targets)

        return z, logits, targets, loss, attn, attn_logits

    def on_train_epoch_start(self):
        self.train_stats['train/num_samples'] = 0
        self.train_stats['class_cnt'] = Counter()

    def on_train_epoch_end(self):
        class_cnt = self.train_stats.pop('class_cnt')
        class_counts = {f'train/class_{k}': v for k, v in class_cnt.items()}
        self.train_stats.update(class_counts)

        self.log_dict(self.train_stats)

        fig, ax = self.cm_train.plot()
        self.cm_train.reset()
        self.logger.experiment.log({'train/confusion_matrix': fig})
        plt.close('all')

        self.train_stats['train/num_samples'] = 0
        self.train_stats['class_cnt'] = Counter()

    def training_step(self, batch, batch_idx):
        # if batch_idx == 0:
        #     torch.save(batch, '/work/FAC/FBM/DBC/mrapsoma/prometex/projects/mesothelioma/workflow-xe-hne/2-mil-end-to-end/train-batch.pt')
        z, logits, targets, loss, attn, attn_logits = self.shared_step(batch)
        batch_size = targets.shape[0]

        # metrics
        preds = torch.argmax(logits, dim=1).long()
        self.train_metrics(logits, targets)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False, batch_size=batch_size)

        self.cm_train(preds, targets)

        # loss
        self.log("loss/train", loss, on_step=True, on_epoch=True, batch_size=batch_size)

        # stats
        self.train_stats['train/num_samples'] += batch_size
        self.train_stats['class_cnt'].update(targets.tolist())

        batch['loss'] = loss
        batch['z'] = z.detach().cpu()
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

        fig, ax = self.cm_val.plot()
        self.cm_val.reset()
        self.logger.experiment.log({'val/confusion_matrix': fig})
        plt.close('all')

        self.val_stats['val/num_samples'] = 0
        self.val_stats['class_cnt'] = Counter()

    def validation_step(self, batch, batch_idx):
        # if batch_idx == 0:
        #     torch.save(batch, '/work/FAC/FBM/DBC/mrapsoma/prometex/projects/mesothelioma/workflow-xe-hne/2-mil-end-to-end/val-batch.pt')
        z, logits, targets, loss, attn, attn_logits = self.shared_step(batch)
        batch_size = targets.shape[0]

        # metrics
        preds = torch.argmax(logits, dim=1).long()
        self.valid_metrics(logits, targets)
        self.log_dict(self.valid_metrics, on_step=False, on_epoch=True, batch_size=batch_size)

        self.cm_val(preds, targets)

        # loss
        self.log("loss/val", loss, on_epoch=True, batch_size=batch_size)

        # stats
        self.val_stats['val/num_samples'] += batch_size
        self.val_stats['class_cnt'].update(targets.tolist())

        batch['loss'] = loss
        batch['z'] = z.detach().cpu()
        return batch

    def configure_optimizers(self):
        optimizer = optim.AdamW([
            {'params': self.head.parameters(), 'lr': self.lr_head},
            {'params': filter(lambda p: p.requires_grad, self.mil.parameters()), 'lr': self.lr_mil},
            {'params': filter(lambda p: p.requires_grad, self.backbone.parameters()), 'lr': self.lr_backbone}
        ], weight_decay=self.weight_decay)

        return optimizer
