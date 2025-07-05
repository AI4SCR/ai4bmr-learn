import torchvision
from pathlib import Path
from torchvision.transforms import v2

class VOCDetection:

    def __init__(self, base_dir: Path | None = None, transform = None, **kwargs):
        base_dir = base_dir or Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/VOCDetection')
        base_dir = base_dir.resolve()
        self.dataset = torchvision.datasets.VOCDetection(root=base_dir, **kwargs)
        self.to_image = v2.ToImage()
        self.transform = transform

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        img = self.to_image(img)

        item = {'image': img, 'target': target, 'index': int(idx)}
        if self.transform is not None:
            item = self.transform(item)

        return item

    def __len__(self):
        return len(self.dataset)