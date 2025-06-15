# %%
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
from loguru import logger
from torch import get_num_threads
from torch.utils.data import DataLoader


@dataclass
class MILDataModuleConfig:
    data_dir: Path
    metadata_dir: Path
    target_column_name: str = "target"
    num_workers: int = None
    persistent_workers: bool = True
    shuffle: bool = True
    pin_memory: bool = True
    random_state: int = 42

from ai4bmr_learn.datamodules.base import BaseMILDataModule
