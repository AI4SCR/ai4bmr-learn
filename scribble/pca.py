# %%
from pathlib import Path
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from ai4bmr_learn.datamodules.dataloader_collection import DataLoader, DataLoaderCollection
from ai4bmr_learn.datasets.coordinates import Coordinates
from ai4bmr_learn.collators.geneformer import GeneformerCollate
from pathlib import Path
from ai4bmr_learn.utils.device import batch_to_device
from ai4bmr_learn.transforms.random_resize_crop import RandomResizeCrop

# transform
transform = RandomResizeCrop(size=224,
                             scale=[1., 1.],
                             ratio=[1., 1.],
                             errors='clip')

# Collator
collate_fn = GeneformerCollate(
    kernel_size=256,
    stride=256,
    model_name="gf-6L-30M-i2048"
)

collate_fn = GeneformerCollate(
    kernel_size=14,
    stride=14,
    model_name="gf-6L-30M-i2048"
)

coords_path = Path(
    "/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/fmx/data/splits/hest1k-tts=4-fvs=0-min_transcripts_per_patch=200/test-0.json")
dataset = Coordinates(coords_path=coords_path, with_image=True,
                      cache_dir=Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/fmx/data/cache'),
                      metadata_path=Path(
                          '/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/fmx/data/datasets/hest1k/metadata.parquet'),
                      index_key='global_id',
                      # transform=transform
                      )
dataset.setup()
item = dataset[0]
item.keys()

# %%

collate_fn([item])

# Single DataLoader
predict_loader = DataLoader(
    dataset=dataset,
    collate_fn=collate_fn,
    batch_size=10
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
