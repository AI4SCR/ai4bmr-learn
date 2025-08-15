from torch.utils.data import Dataset
from pathlib import Path
import torch
from loguru import logger
from typing import Callable

class Embeddings(Dataset):

    def __init__(self, embeddings_dir: Path, transform: Callable | None = None, pattern: str = '*.pt'):
        super().__init__()

        self.embeddings_dir = embeddings_dir.resolve()
        assert self.embeddings_dir.exists(), f'{self.embeddings_dir} does not exist.'
        self.pattern = pattern

        self.transform = transform
        self.embeddings: list | None = None

    def setup(self):
        embedding_paths = sorted(self.embeddings_dir.glob(self.pattern))
        logger.info(f'Found {len(embedding_paths)} embeddings in {self.embeddings_dir}')

        self.embeddings = []
        for embedding_path in embedding_paths:
            embedding = torch.load(str(embedding_path), map_location='cpu', weights_only=True)
            if self.transform is not None:
                embedding = self.transform(embedding)
            self.embeddings.append(embedding)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        return self.embeddings

# ds = Embeddings(Path('/users/amarti51/prometex/data/benchmarking/embeddings/PCa/oee8hrej'))
# ds.setup()
