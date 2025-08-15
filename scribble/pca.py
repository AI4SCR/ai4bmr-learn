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


# %%
from ai4bmr_learn.datamodules.dataloader_collection import DataLoader, DataLoaderCollection
from ai4bmr_learn.datasets.coordinates import Coordinates
from ai4bmr_learn.collators.geneformer import GeneformerCollate
from pathlib import Path
from ai4bmr_learn.utils.device import batch_to_device

# Collator
collate_fn = GeneformerCollate(
    kernel_size=256,
    stride=256,
    model_name="gf-6L-30M-i2048"
)

collate_fn = GeneformerCollate(
    kernel_size=16,
    stride=16,
    model_name="gf-6L-30M-i2048"
)

coords_path=Path("/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/fmx/data/splits/hest1k-tts=4-fvs=0-min_transcripts_per_patch=200/test-0.json")
dataset = Coordinates(coords_path=coords_path, with_image=False,
                      cache_dir = Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/fmx/data/cache'))
dataset.setup()
item = dataset[0]
item.keys()

# Single DataLoader
predict_loader = DataLoader(
    dataset=dataset,
    collate_fn=collate_fn,
    batch_size = 10
)

# Collection of DataLoaders (only predict in this case)
dataloader_collection = DataLoaderCollection(
    dataloaders={"predict": [predict_loader]}
)

batch = next(iter(predict_loader))
batch['expression']['input_ids'].shape
batch = batch_to_device(batch, 'cuda')

from ai4bmr_learn.modules.model_builder import ModelBuilder
model = ModelBuilder(path='ai4bmr_learn.models.encoder.geneformer.Geneformer', as_kwargs=True, batch_key='expression')
model.to('cuda')
out = model(batch)

# %
# import shutil
# from tqdm import tqdm
# from pathlib import Path
# cache_dir = Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/fmx/data/cache')
# cache_dir.mkdir(parents=True, exist_ok=True)
# cache_paths = Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/fmx/data/datasets/hest1k').rglob('patches/*.parquet')
# for cache_path in tqdm(cache_paths):
#     target_path = cache_dir / cache_path.name
#     shutil.move(cache_path, target_path)
