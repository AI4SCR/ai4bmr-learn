import lightning as L
import torch
import torch.nn as nn
from glom import glom
from importlib import import_module
from inspect import isclass, signature
from typing import Any, Callable
from pathlib import Path

def resolve_dot_path(path: str) -> Callable[..., Any]:
    """Resolve 'pkg.module.func' to the actual callable."""
    module_name, attr_name = path.rsplit('.', 1)
    module = import_module(module_name)
    return getattr(module, attr_name)

def build_object(path: str, *args, **kwargs) -> Any:
    """
    Import a function or class from a dot-path and call/instantiate it.

    Args:
        path: Dot-path to a callable (function or class).
        strict: If True, raise error for unexpected keyword arguments.
        *args/**kwargs: Arguments to pass to the callable.
    """
    obj = resolve_dot_path(path)
    return obj(*args, **kwargs)


class ModelBuilder(L.LightningModule):

    def __init__(self, path: str, batch_key: str | None = None, as_kwargs: bool = False, **kwargs):
        super().__init__()
        self.model = build_object(path=path, **kwargs)
        self.batch_key = batch_key
        self.as_kwargs = as_kwargs
        self.save_hyperparameters()

    def forward(self, x) -> Any:
        x = glom(x, self.batch_key) if self.batch_key else x
        return self.model(**x) if self.as_kwargs else self.model(x)


class Model(L.LightningModule):

    def __init__(self, model = nn.Module, batch_key: str | None = None, as_kwargs: bool = False, ckpt_path: Path | None = None, **kwargs):
        super().__init__()
        self.model = model
        self.batch_key = batch_key
        self.as_kwargs = as_kwargs

        self.ckpt_path = ckpt_path
        if self.ckpt_path is not None:
            raise NotImplementedError()
        #     ckpt = torch.load(ckpt_path, map_location="cpu")
        #     state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        #     model.load_state_dict(state_dict=state_dict, strict=True)
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x) -> Any:
        x = glom(x, self.batch_key) if self.batch_key else x
        return self.model(**x) if self.as_kwargs else self.model(x)