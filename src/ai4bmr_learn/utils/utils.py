from typing import Iterable, Iterator, TypeVar, List
from itertools import islice

T = TypeVar('T')

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def batched(iterable, n: int):
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]

def chunked(iterable: Iterable[T], n: int) -> Iterator[List[T]]:
    """Split iterable into successive non-overlapping chunks of size n."""
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk

def get_batch_size(iterable, n: int):
    return (len(iterable) + n - 1) // n