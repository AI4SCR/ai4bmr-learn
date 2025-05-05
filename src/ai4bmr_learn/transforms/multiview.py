import torch.nn as nn

class MultiViewTransform(nn.Module):

    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self, item):
        views = [transform(item) for transform in self.transforms]
        return dict(views=views, item=item)