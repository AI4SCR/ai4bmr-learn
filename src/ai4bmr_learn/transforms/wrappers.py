from torchvision.transforms import v2
import pandas as pd
from pathlib import Path

class Normalize(v2.Normalize):

    def __init__(self, stats_path: str):
        stats_path = Path(stats_path)
        assert stats_path.exists()

        stats = pd.read_json(stats_path)
        super().__init__(mean=stats['mean'], std=stats['std'])

# Normalize(stats_path='/users/amarti51/prometex/data/dinov1/datasets/Cords2024/images/default/stats.json')
