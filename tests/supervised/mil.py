import torch
import torch.nn as nn
import torch.nn.functional as F
from ai4bmr_learn.models.backbones.timm import Backbone
from torch.utils.data import DataLoader

def test_mil():
    from ai4bmr_learn.supervised.mil import MIL

    class Head(nn.Module):
        def __init__(self, input_dim: int = 64, num_classes: int = 2):
            super().__init__()
            self.head = nn.Linear(input_dim, num_classes)

        def forward(self, x):
            return self.head(x.mean(dim=0, keepdim=True))

    class SimpleBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=16, stride=16)
            self.pool = nn.AdaptiveAvgPool2d(8)

        def forward(self, x):
            x = self.conv(x)
            x = F.relu(x)
            x = self.pool(x)
            return x.flatten(start_dim=1)

    bag = {'instances': torch.randn(2, 3, 224, 224), 'target': 0}
    backbone, input_dim = SimpleBackbone(), 64
    backbone, input_dim = Backbone(model_name='resnet18', global_pool='avg'), 512
    head = Head(input_dim=input_dim)

    z = backbone(bag['instances'])
    head(z).shape

    bag1 = {'instances': torch.randn(2, 3, 224, 224), 'target': 0}
    bag2 = {'instances': torch.randn(2, 3, 224, 224), 'target': 1}
    dl = DataLoader([bag1, bag2], batch_size=2)
    batch = next(iter(dl))

    mil = MIL(backbone=backbone, head=head, num_classes=2, batch_key='instances', target_key='target')
    mil.shared_step(batch)
