import torch
import torchvision.transforms.functional as F

class PadToMinSize:
    def __init__(self, min_size: int, fill=0):
        self.min_size = min_size
        self.fill = fill

    def __call__(self, img: torch.Tensor):
        h, w = img.shape[-2:]
        pad_h = max(0, self.min_size - h)
        pad_w = max(0, self.min_size - w)
        # (left, top, right, bottom)
        padding = (0, 0, pad_w, pad_h)
        return F.pad(img, padding, fill=self.fill)

