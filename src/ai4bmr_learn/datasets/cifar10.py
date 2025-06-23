import torchvision
from pathlib import Path
from torchvision.tv_tensors import Image
import torch

class CIFAR10(torchvision.datasets.CIFAR10):

    def __init__(self, base_dir: Path | None = None, **kwargs):
        base_dir = base_dir or Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/cifar10')
        base_dir = base_dir.resolve()
        super().__init__(root=base_dir, **kwargs)

    def setup(self, *args, **kwargs):
        pass

    def __getitem__(self, idx):
        image, target = self.data[idx], self.targets[idx]
        item = {'image': image, 'target': target}
        if self.transform is not None:
            item = self.transform(item)
        return item

