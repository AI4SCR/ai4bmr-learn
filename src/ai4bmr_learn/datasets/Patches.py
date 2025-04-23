from torch.utils.data import Dataset
from dataclasses import asdict
from ai4bmr_learn.data_models.Coordinate import BaseCoordinate, SlideCoordinate
from ai4bmr_learn.utils.utils import pair
from ai4bmr_learn.utils.images import get_patch
import openslide
import torch

class Patches(Dataset):
    def __init__(self, coords: list[BaseCoordinate | SlideCoordinate], transform=None):
        super().__init__()
        self.coords = coords
        self.transform = transform

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]

        patch = get_patch(coord)
        
        item = {**asdict(coord)}
        item["image"] = patch

        if self.transform:
            item = self.transform(item)
        return item