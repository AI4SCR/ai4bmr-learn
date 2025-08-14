from importlib import import_module
from inspect import isclass, signature
from typing import Any, Callable

def resolve_dot_path(path: str) -> Callable[..., Any]:
    """Resolve 'pkg.module.func' to the actual callable."""
    module_name, attr_name = path.rsplit('.', 1)
    module = import_module(module_name)
    return getattr(module, attr_name)

def _filter_kwargs(callable_obj: Callable, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter out keyword arguments the callable doesn’t accept, unless it has **kwargs."""
    sig = signature(callable_obj)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

def build_object(path: str, *args, **kwargs) -> Any:
    """
    Import a function or class from a dot‑path and call/instantiate it with the given args/kwargs.
    """
    obj = resolve_dot_path(path)
    kwargs = _filter_kwargs(obj, kwargs)
    return obj(*args, **kwargs)

# Instantiate a timm ResNet18 (pretrained on ImageNet) if timm is installed:
model = build_object("timm.create_model", model_name="resnet18", pretrained=True)
