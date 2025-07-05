import torchvision
from pathlib import Path
from torchvision.transforms import v2
from torch.utils.data import Dataset

class ImageNet(Dataset):

    def __init__(self, base_dir: Path | None = None, transform = None, **kwargs):
        super().__init__()

        base_dir = base_dir or Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/imagenet')
        base_dir = base_dir.resolve()
        self.dataset = torchvision.datasets.ImageNet(root=base_dir, **kwargs)
        self.to_image = v2.ToImage()
        self.transform = transform
        self.idx_to_class = {v:k for k,v in self.dataset.class_to_idx.items()}

    def __getitem__(self, idx):
        path, target = self.dataset.samples[idx]
        img = self.dataset.loader(path)
        class_ = self.idx_to_class[target]

        img = self.to_image(img)

        item = {'image': img, 'target': target, 'class': class_, 'index': int(idx)}
        if self.transform is not None:
            item = self.transform(item)

        return item

    def __len__(self):
        return len(self.dataset)