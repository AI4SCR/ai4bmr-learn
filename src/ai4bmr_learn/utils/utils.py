from typing import Iterable, Iterator, TypeVar, List
from itertools import islice

from jsonargparse import Namespace

T = TypeVar('T')


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def batched(iterable: Iterable[T], n: int) -> Iterator[List[T]]:
    """Split iterable into successive non-overlapping batches of size n."""
    assert n > 0, "n must be > 0"
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk


def chunked(iterable: Iterable[T], n: int) -> Iterator[List[T]]:
    """Split an iterable into n chunks, trying to balance their lengths."""
    assert n > 0, "n must be > 0"
    items = list(iterable)
    k, m = divmod(len(items), n)
    chunk_sizes = [k] * (n - 1) + [k + m]

    idx = 0
    for size in chunk_sizes:
        yield items[idx : idx + size]
        idx += size


def get_batch_size(iterable, n: int):
    assert n > 0, "n must be > 0"
    return (len(iterable) + n - 1) // n


def setup_wandb_auth(api_key_name: str = "WANDB_API_KEY_ETHZ"):
    import os
    import netrc
    from pathlib import Path

    api_key = os.getenv(api_key_name)
    if api_key is None:
        netrc_path = Path.home() / ".netrc"
        assert netrc_path.exists(), f"No API key in {api_key_name} and missing {netrc_path}"
        auth = netrc.netrc(str(netrc_path)).authenticators("api.wandb.ai")
        assert auth is not None and auth[2], "Missing api.wandb.ai credentials in ~/.netrc"
        api_key = auth[2]

    os.environ["WANDB_API_KEY"] = api_key


def to_dict(item):
    item = vars(item) if isinstance(item, Namespace) else item

    if isinstance(item, dict):
        return {k: to_dict(v) for k, v in item.items()}

    if isinstance(item, (list, tuple)):
        return [to_dict(i) for i in item]

    return item
