import lightning as L
from ai4bmr_learn.callbacks.log_trainer import LogTrainer
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule

log_trainer = LogTrainer()
trainer = L.Trainer(callbacks=[log_trainer])
trainer.fit(DemoModel(), BoringDataModule())