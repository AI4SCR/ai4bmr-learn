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
ds = PointsPatches(coords=coords)
item = ds[0]

# %%
from ai4bmr_learn.transforms.random_crop import RandomCrop
from ai4bmr_learn.transforms.multiview import MultiViewTransform

global_transform = [RandomCrop(scale=(0.66, 1)), RandomCrop(scale=(0.66, 1))]
local_transform = [RandomCrop(scale=(0.25, 0.66)), RandomCrop(scale=(0.25, 0.66)), RandomCrop(scale=(0.25, 0.66)), RandomCrop(scale=(0.25, 0.66))]
views_transform = global_transform + local_transform
transform = MultiViewTransform(transforms=views_transform)

ds = PointsPatches(coords=coords, transform=transform)
item = ds[0]

for view in item['views']:
    print(view['data'].shape)
    print(view['metadata']['coord']['height'])
    print(view['metadata']['coord']['width'])

# %% MODEL
import torch.nn as nn
import torch
class PointsEncoder(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_channels))

        layer = nn.TransformerEncoderLayer(d_model=in_channels, nhead=4, dim_feedforward=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)

    def forward(self, x):
        x = torch.cat([self.cls_token, x], dim=1)
        return self.encoder(x)

backbone = PointsEncoder(in_channels=num_features)
inp = torch.randn((1, 10, num_features))
inp = item['views'][4]['data'].unsqueeze(0)
out = backbone(inp)

# %% SSL-TRAINING
from ai4bmr_learn.ssl.dino_light import DINOLight
ssl = DINOLight(backbone=backbone, input_dim=16, hidden_dim=16, output_dim=32)