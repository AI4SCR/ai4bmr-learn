from typing import Iterable, Iterator, TypeVar, List
from itertools import islice

T = TypeVar('T')


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def batched(iterable: Iterable[T], n: int) -> Iterator[List[T]]:
    """Split iterable into successive non-overlapping batches of size n."""
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk


def chunked(iterable: Iterable[T], n: int) -> List[List[T]]:
    """Split an iterable into n chunks, trying to balance their lengths."""
    items = list(iterable)
    k, m = divmod(len(items), n)
    n += m > 0
    return [items[i * k:(i + 1) * k] for i in range(n)]


def get_batch_size(iterable, n: int):
    return (len(iterable) + n - 1) // n
