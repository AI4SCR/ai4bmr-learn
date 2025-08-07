import lightning as L
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, Recall


class TestLogging(L.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1,1)
        self.rgn = np.random.default_rng(0)
        self.train_values = self.rgn.random(size=(100,))
        self.val_values = self.rgn.random(size=(100,))

        self.train_acc = Accuracy(num_classes=3, average='macro', task='multiclass')
        self.train_recall = Recall(num_classes=3, average='macro', task='multiclass')

        self.val_acc = Accuracy(num_classes=3, average='macro', task='multiclass')
        self.val_recall = Recall(num_classes=3, average='macro', task='multiclass')

        self.train_preds = torch.randint(0,3, size=(100, 3))
        self.train_targets = torch.randint(0,3, size=(100, 3))

        self.val_preds = torch.randint(0,3, size=(100, 3))
        self.val_targets = torch.randint(0,3, size=(100,3 ))

    def training_step(self, batch, batch_idx):
        val = self.train_values[batch_idx]

        preds = self.train_preds[batch_idx]
        targets = self.train_targets[batch_idx]

        val1 = Accuracy(num_classes=3, average='macro', task='multiclass')(preds, targets)
        val2 = self.train_acc(preds=preds, target=targets)

        self.log('train_val1', val1, on_step=True)
        self.log('train_val2', val2, on_step=True)
        self.log('train_acc', self.train_acc, on_step=True)

    def validation_step(self, batch, batch_idx):
        val = self.val_values[batch_idx]

        preds = self.val_preds[batch_idx]
        targets = self.val_targets[batch_idx]

        val1 = Accuracy(num_classes=3, average='macro', task='multiclass')(preds, targets)
        val2 = self.val_acc(preds=preds, target=targets)

        self.log('val_val1', val1, on_step=False)
        self.log('val_val2', val2, on_step=False)
        self.log('val_acc', self.val_acc, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

model = TestLogging()

dl_train = DataLoader([1] * 100, batch_size=1)
dl_val = DataLoader([1] * 100, batch_size=1)

logger = CSVLogger(save_dir='/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ai4bmr-learn/tests/supervised')
trainer = L.Trainer(
    accelerator='cpu',
    # log_every_n_steps=101,
    # validate_every_n_epochs=1,
    # val_check_interval=.2,
    # val_check_interval=None,
    logger=logger,
    max_epochs=2
)
trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_val)

acc = []
from sklearn.metrics import balanced_accuracy_score, recall_score
p, t = [], []
bs = []
recall = []
for pred, targets in zip(model.val_preds, model.val_targets):
    p.extend(pred.tolist())
    t.extend(targets.tolist())
    bs.append(Accuracy(num_classes=3, average='macro', task='multiclass')(pred, targets))
    recall.append(recall_score(targets.tolist(), pred.tolist(), average='macro'))
    # bs.append(Accuracy(targets, pred))

balanced_accuracy_score(t, p)
recall_score(t, p, average='macro')
np.mean(bs)
np.mean(recall)
import pandas as pd
df = pd.read_csv('/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ai4bmr-learn/tests/supervised/lightning_logs/version_0/metrics.csv')
df[df.train_acc_epoch.notna()]
df[df.val_acc_step.notna()]
model.val_values[:3]
np.mean([model.val_values])
np.mean([model.train_values[49], model.train_values[99]])

# NOTE:
# on_step: log single batch value at step x
# on_epoch: aggregates all values seen in an epoch, not just at the log steps