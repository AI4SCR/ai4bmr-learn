import torchvision
from pathlib import Path

class CIFAR10(torchvision.datasets.CIFAR10):

    def __init__(self, base_dir: Path | None = None, **kwargs):
        base_dir = base_dir or Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/cifar10')
        base_dir = base_dir.resolve()
        super().__init__(root=base_dir, **kwargs)

    def __getitem__(self, item):
        image, target = super().__getitem__(item)
        return {'image': image, 'target': target}

