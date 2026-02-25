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

def to_dict(item):
    item = vars(item) if isinstance(item, Namespace) else item

    if isinstance(item, dict):
        return {k: to_dict(v) for k, v in item.items()}

    if isinstance(item, (list, tuple)):
        return [to_dict(i) for i in item]

    return item
