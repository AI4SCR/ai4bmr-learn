import torch
from abc import abstractmethod

from pydantic import BaseModel, Field, ConfigDict


class DecoderDefault(BaseModel):
    num_tokens: int = None
    dim: int = 192
    num_layers: int = 4
    num_heads: int = 3


class DecoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "default"
    init_kwargs: DecoderDefault = Field(default_factory=DecoderDefault)


class BaseDecoder(torch.nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class BaseMaskedDecoder(torch.nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def forward_masked(self, x: torch.Tensor, idx_keep: torch.Tensor) -> torch.Tensor:
        pass
