import lightning as L
from torch.utils.data import DataLoader
from torch import get_num_threads
import torch
from ai4bmr_learn.transforms.dino_transform import DINOTransform
from torch.utils.data import Subset
from torchvision.transforms import v2
from lightly.transforms.dino_transform import IMAGENET_NORMALIZE

class ImageNet(L.LightningDataModule):

    def __init__(self,
                 batch_size: int = 64,
                 num_workers: int = None,
                 persistent_workers: bool = True,
                 shuffle: bool = True,
                 pin_memory: bool = True
                 ):
        super().__init__()

        # DATALOADERS
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else max(0, get_num_threads() - 1)
        self.persistent_workers = persistent_workers if self.num_workers > 0 else False
        self.shuffle = shuffle
        self.pin_memory = pin_memory

        # DATASET
        num_classes = 10
        max_index = 1300 * num_classes

        g = torch.Generator().manual_seed(0)
        indices = torch.randperm(max_index, generator=g)

        self.train_indices = indices[:int(max_index * 0.9)]
        self.val_indices = indices[int(max_index * 0.9):]
        self.test_idx = None

        self.train_set = self.val_set = self.test_set = None

        # TRANSFORMS
        self.train_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            DINOTransform(),
            v2.Normalize(mean=IMAGENET_NORMALIZE['mean'], std=IMAGENET_NORMALIZE['std'])
        ])

        self.val_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224)),
            v2.Normalize(mean=IMAGENET_NORMALIZE['mean'], std=IMAGENET_NORMALIZE['std'])
        ])

        self.save_hyperparameters()

    def setup(self, stage):
        from ai4bmr_learn.datasets.imagenet import ImageNet
        self.train_set = Subset(ImageNet(transform=self.train_transform), indices=self.train_indices)
        self.val_set = Subset(ImageNet(transform=self.val_transform), indices=self.val_indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

