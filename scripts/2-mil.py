import torchvision.transforms.v2

from ai4bmr_learn.datasets.mil import MILFromDataset
from ai4bmr_learn.supervised.mil import MIL
from ai4bmr_learn.models.backbones.timm import Backbone
from ai4bmr_learn.models.mil.linear import Linear
from torch.utils.data import DataLoader, Dataset
from ai4bmr_learn.datasets.items import Items
import torch
from pathlib import Path

class DummyDataset(Dataset):

    def __init__(self):
        self.bag_ids = torch.randint(0, 35, size=(100,)).tolist()

    def __getitem__(self, idx):
        return dict(image=torch.randn(3, 224, 224), target=0 if torch.randn(1).item() > 0.5 else 0)

    def __len__(self):
        return len(self.bag_ids)

dataset_dir = Path('/users/amarti51/prometex/data/benchmarking/datasets/Cords2024')
dataset_dir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/omics-embed/datasets/Dannenberg2022')
transform = torchvision.transforms.v2.ToDtype(torch.float32, scale=False)
ds = Items(items_path=dataset_dir / 'coords' / 'size=224-stride=224.json',
             metadata_path=dataset_dir / 'splits' / 'coords' / 'clf-target=dx_name.parquet',
             split='fit', drop_nan_columns=True, transform=transform)
ds.setup()
item = ds[0]

# ds = DummyDataset()
mil_ds = MILFromDataset(dataset=ds, num_instances=4, pad=True, bag_ids_attr='image_ids')
mil_ds.setup()
bag = mil_ds[0]

num_classes = 3
backbone = Backbone(model_name='resnet18', global_pool='avg', num_channels=43)
head = Linear(input_dim=512, num_classes=num_classes)
mil = MIL(backbone=backbone, head=head, num_classes=num_classes, target_key='metadata.dx_name')

dl = DataLoader([mil_ds[0], mil_ds[2]], batch_size=2)
batch = next(iter(dl))
outs = mil.training_step(batch, -1)
