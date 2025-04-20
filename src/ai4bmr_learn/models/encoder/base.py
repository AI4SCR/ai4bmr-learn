import torch
from abc import abstractmethod
from pydantic import BaseModel, Field, ConfigDict

class EncoderDefault(BaseModel):
    dim: int = 192
    num_layers: int = 12
    num_heads: int = 3
    num_prefix_tokens: int = 1


class EncoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "default"
    init_kwargs: EncoderDefault = Field(default_factory=EncoderDefault)


class BaseEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass


class BaseMaskedEncoder(BaseEncoder):
    def __init__(self, num_tokens: int):
        super().__init__()
        self.num_tokens = num_tokens

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def forward_masked(self, x, idx_keep: torch.Tensor):
        pass