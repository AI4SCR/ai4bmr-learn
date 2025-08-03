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
    chunk_sizes = [k] * (n-1) + [k + m]

    idx = 0
    while chunk_sizes:
        size = chunk_sizes.pop(0)
        yield items[idx:idx+size]
        idx += size



def get_batch_size(iterable, n: int):
    return (len(iterable) + n - 1) // n


def setup_wandb_auth(api_key_name: str = 'WANDB_API_KEY_ETHZ'):
    import os
    import netrc

    # Try environment variable first
    api_key = os.getenv(api_key_name)

    if api_key is None:
        try:
            # Default wandb API host
            machine = "api.wandb.ai"
            auth = netrc.netrc().authenticators(machine)
            if auth:
                login, _, api_key = auth
        except FileNotFoundError:
            print("No .netrc file found")
        except Exception as e:
            print(f"Failed to load API key from .netrc: {e}")

    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
    else:
        raise RuntimeError("No WANDB API key found in env or .netrc")
