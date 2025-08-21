import lightning as L
from ai4bmr_learn.logging.loggers import WandbLogger
from ai4bmr_learn.callbacks.save_transforms import SaveTransforms
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule
import torch

class DummyDataset:
    transform = v2.Compose([v2.ToTensor()])

    def __len__(self):
        return 50

    def __getitem__(self, idx):
        return torch.randn((1, 32))

class DummyModel(DemoModel):

    def validation_step(self, batch, batch_idx, dataloader_idx: int = None):
        return self.training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx, dataloader_idx: int = None):
        return self.training_step(batch, batch_idx)

model = DummyModel()
loaders = BoringDataModule()

ds = DummyDataset()
dl = DataLoader(ds)

logger = WandbLogger(save_dir='/users/amarti51/prometex/data/logs')
_  = logger.experiment  # init run

save_dir = Path(logger.save_dir) / logger.experiment.project / logger.experiment._attach_id / 'transforms'
save_transforms = SaveTransforms()

trainer = L.Trainer(callbacks=[save_transforms], logger=logger, max_epochs=1, num_sanity_val_steps=1)
trainer.fit(model=model, train_dataloaders=dl, val_dataloaders=[dl, dl])
trainer.test(model=model, dataloaders=[dl, dl])
# trainer.validate(model=model, dataloaders=dl)
# save_transforms.on_fit_start(trainer=trainer, pl_module=DemoModel)