from torch.utils.data import Dataset
from pathlib import Path
import torch
from loguru import logger
from typing import Callable

class Embeddings(Dataset):

    def __init__(self, data_dir: Path | None = None, transform: Callable | None = None, pattern: str = '*.pt'):
        super().__init__()
        
        self.data_dir = data_dir.resolve()
        assert self.data_dir.exists(), f'{self.data_dir} does not exist.'
        self.pattern = pattern

        self.transform = transform
        self.data_paths: list[Path] | None = None

    def setup(self):
        self.data_paths = sorted(self.data_dir.glob(self.pattern))
        logger.info(f'Found {len(self.data_paths)} embeddings in {self.data_dir}')

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, idx: int):
        data_path = self.data_paths[idx]
        embedding = torch.load(str(data_path), map_location='cpu', weights_only=True)
        return embedding

# ds = Embeddings(Path('/users/amarti51/prometex/data/benchmarking/embeddings/PCa/oee8hrej'))
# ds.setup()
