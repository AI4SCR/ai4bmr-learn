# %%
from ai4bmr_datasets.datasets.DummyPoints import DummyPoints
from ai4bmr_learn.datasets.PointsPatches import PointsPatches

from ai4bmr_learn.utils.images import get_coordinates_dict
from ai4bmr_learn.data_models.Coordinate import PointsCoordinate

# %%
height = width = 224
ds = DummyPoints(height=height, width=width, num_features=16)
ds.prepare_data(force=False)
ds.setup()

num_features = ds.num_features
item = ds[0]

# %%
coord_dicts = get_coordinates_dict(height=ds.height, width=ds.width, kernel_size=32, stride=32)
coords = [PointsCoordinate(**i, points_path=str(ds.data_dir / f'{sample_id}.gpkg'))
          for sample_id in ds.sample_ids for i in coord_dicts]

coord = coords[0]
labels = ['type_1', 'type_2', 'type_3', 'type_4']
ds = PointsPatches(coords=coords, labels=labels, label_key='label')
item = ds[0]

# %%
from ai4bmr_learn.transforms.random_crop import RandomCrop
from ai4bmr_learn.transforms.multiview import MultiViewTransform
from torch.utils.data import DataLoader

global_transform = [RandomCrop(scale=(0.66, 1)), RandomCrop(scale=(0.66, 1))]
local_transform = [RandomCrop(scale=(0.25, 0.66)), RandomCrop(scale=(0.25, 0.66)), RandomCrop(scale=(0.25, 0.66)), RandomCrop(scale=(0.25, 0.66))]
views_transform = global_transform + local_transform
transform = MultiViewTransform(transforms=views_transform)

ds = PointsPatches(coords=coords, labels=labels, transform=transform)
item = ds[0]
dl = DataLoader(ds, batch_size=5)
batch = next(iter(dl))

# for view in item['views']:
#     print(view['data'].shape)
#     print(view['metadata']['coord']['height'])
#     print(view['metadata']['coord']['width'])

# %% MODEL
import torch.nn.functional as F
import torch.nn as nn
import torch
class PointsEncoder(nn.Module):

    def __init__(self, in_channels: int, embed_dim: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.proj = nn.Linear(in_channels, embed_dim)

    def forward(self, x):
        return F.relu(self.proj(x))

backbone = PointsEncoder(in_channels=len(labels))
inp = torch.randn((1, 10, len(labels)))
inp = item['views'][4]['data'].unsqueeze(0)
out = backbone(inp)

# %% SSL-TRAINING
import lightning as L
from ai4bmr_learn.ssl.dino_light import DINOLight
ssl = DINOLight(backbone=backbone, input_dim=4, hidden_dim=4, output_dim=4, view_key='data')
ssl.training_step(batch, batch_idx=0)

trainer = L.Trainer(max_epochs=50, devices=1)
trainer.fit(model=ssl, train_dataloaders=dl)

# %%