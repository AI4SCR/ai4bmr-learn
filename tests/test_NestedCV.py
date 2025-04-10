from ai4bmr_learn.datamodules.NestedCV import NestedCV


def test_nested_cv():
    import pandas as pd

    from pathlib import Path

    save_dir = Path("~/data/ai4bmr-learn/tests/nested_cv").expanduser().resolve()
    num_samples = 100
    metadata = pd.DataFrame(index=range(num_samples))
    ncv = NestedCV(metadata=metadata, save_dir=save_dir, test_size=0.2, num_outer_cv=5, num_inner_cv=5)

    fold0 = ncv.get_dataset(outer_fold=0)
    assert len(fold0) == num_samples

    inner_fold = ncv.get_dataset(outer_fold=0, inner_fold=0)
    assert len(inner_fold) == num_samples * 0.8
