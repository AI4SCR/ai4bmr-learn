from torch.utils.data import Dataset
from dataclasses import asdict
from ai4bmr_learn.data_models.Coordinate import BaseCoordinate
from ai4bmr_learn.utils.utils import pair
from ai4bmr_learn.utils.images import get_patch
from torchvision.tv_tensors import Image
import openslide
import torch

class Patches(Dataset):
    def __init__(self, coords: list[BaseCoordinate], transform=None):
        super().__init__()
        self.coords = coords
        self.transform = transform

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        
        item = {**asdict(coord)}

        img_path = item["image_path"]
        slide = openslide.OpenSlide(img_path)

        x, y = item["x"], item["y"]
        patch_height, patch_width = pair(item["kernel_size"])

        patch = slide.read_region((x, y), 0, (patch_width, patch_height))
        patch = patch.convert("RGB")

        item["patch"] = Image(patch)

        if self.transform:
            item = self.transform(item)
        return item