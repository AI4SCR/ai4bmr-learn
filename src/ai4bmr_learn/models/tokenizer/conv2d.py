import torch
import torch.nn as nn
from einops import rearrange
from ai4bmr_learn.models.tokenizer.base import BaseTokenizer

class TokenizerConv(BaseTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.proj = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            self.image_size == x.shape[2:]
        ), f"image size mismatch, expected {self.image_size} but got {x.shape[2:]}"

        x = self.proj(x)
        x = rearrange(x, f"b d h w -> b (h w) d")
        return x

    def tokens2img(self, x: torch.Tensor) -> torch.Tensor:
        kh, kw = self.kernel_size
        h, w = self.grid_size
        img = rearrange(x, "b (h w) (c kh kw) -> b c (h kh) (w kw)", kh=kh, kw=kw, h=h, w=w)
        return img