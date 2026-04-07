from pathlib import Path

import pandas as pd
import pytest

from ai4bmr_learn.data.splits import save_splits


def test_save_splits_rejects_zero_val_size(tmp_path: Path):
    metadata = pd.DataFrame(index=range(10))

    with pytest.raises(AssertionError, match="use `None` to disable inner splits"):
        save_splits(
            metadata=metadata,
            save_dir=tmp_path / "splits",
            test_size=0.2,
            val_size=0,
        )
