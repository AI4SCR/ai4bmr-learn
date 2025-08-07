import torch
import torch.nn as nn
import torch.nn.functional as F
from ai4bmr_learn.models.backbones.timm import Backbone
from torch.utils.data import DataLoader
import torch
from ai4bmr_learn.datasets.mil import MILFromDataset
from torch.utils.data import Dataset, DataLoader
import tempfile
from pathlib import Path

def test_mil():
    from ai4bmr_learn.supervised.mil import MIL

    class DummyDataset(Dataset):

        def __init__(self):
            self.bag_ids = torch.randint(0,3, size=(100,)).tolist()

        def __getitem__(self, idx):
            return dict(image=torch.randn(3, 224, 224), target=0 if torch.randn(1).item() > 0.5 else 0)

        def __len__(self):
            return len(self.bag_ids)

    ds = DummyDataset()

    ds_mil = MILFromDataset(dataset=ds, num_instances=5)
    ds_mil.setup()

    class Head(nn.Module):
        def __init__(self, input_dim: int = 64, num_classes: int = 2):
            super().__init__()
            self.head = nn.Linear(input_dim, num_classes)

        def forward(self, x):
            x = x.mean(dim=1)  # B, M, D
            return self.head(x)

    backbone, input_dim = Backbone(model_name='resnet18', global_pool='avg'), 512
    head = Head(input_dim=input_dim)

    bag1 =  ds_mil[0]
    z = backbone(bag1['image'])
    z = z.unsqueeze(0)
    head(z).shape

    bag2 = ds_mil[1]
    dl = DataLoader([bag1, bag2], batch_size=2)
    batch = next(iter(dl))

    mil = MIL(backbone=backbone, head=head, num_classes=2, batch_key='image', target_key='target')
    mil.shared_step(batch)

    # %%
test_mil()