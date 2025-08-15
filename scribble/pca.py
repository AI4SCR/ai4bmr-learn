from pathlib import Path
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from ai4bmr_learn.models.classical.logistic_regression import LogisticRegression
from ai4bmr_learn.datamodules.dataloader_collection import DataLoaderCollection, DataLoader
from ai4bmr_learn.datasets.embeddings import Embeddings
from pathlib import Path

model = LogisticRegression(
    batch_key="z",
    target_key="metadata.disease_progr",
    num_classes=2
)

embeddings_dir = Path("/users/amarti51/prometex/data/benchmarking/embeddings/PCa/oee8hrej")

ds_fit = Embeddings(embeddings_dir=embeddings_dir / 'fit')
ds_fit.setup()
ds_val = Embeddings(embeddings_dir=embeddings_dir / 'test')

dl_fit = DataLoader(dataset=ds_fit)
batch = next(iter(dl_fit))
dl_val = DataLoader(dataset=ds_val)

datamodule = DataLoaderCollection(
    dataloaders={
        "fit": [dl_fit],
        "val": [dl_val],
    }
)

trainer = L.Trainer(accelerator='cpu', logger=None, fast_dev_run=False)

trainer.fit(model=model, datamodule=datamodule)
# optional: run predict on the 'predict' split the config defined
# preds = trainer.predict(model=model, datamodule=datamodule)
# do whatever you want with `preds` (list of step outputs)
