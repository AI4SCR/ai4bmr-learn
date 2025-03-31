# %%
from pathlib import Path

from .Tabular import TabularDataModule


class PCaCellFrequenciesDataModule(TabularDataModule):

    def __init__(
        self,
        data_path: Path,
        metadata_path: Path,
        target_column_name: str = "cause_of_death",
        splits_path: Path = None,
        **kwargs
    ):
        super().__init__(
            data_path=data_path,
            metadata_path=metadata_path,
            splits_path=splits_path,
            target_column_name=target_column_name,
            **kwargs
        )

    def _prepare_data(self) -> None:
        # NOTE: here we load one of our datasets and bring it into the right format for the training that we want to do.
        from ai4bmr_datasets.datasets.PCa import PCa
        from pathlib import Path

        ds = PCa(base_dir=Path("/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/PCa"))
        ds.process()
        data = ds.load(image_version="filtered", mask_version="cleaned", features_as_published=False)

        import pandas as pd

        def compute_label_frequency(data: pd.DataFrame, level: str, log_scale: bool = True) -> pd.DataFrame:
            if log_scale:
                pdat = data.groupby(["sample_id"])[level].value_counts()
                pdat += 1
                pdat /= pdat.groupby(["sample_id"]).sum()
                pdat.name = "proportion"
            else:
                pdat = data.groupby(["sample_id"])[level].value_counts(normalize=True)

            return pdat

        metadata = data["samples"][["cause_of_death", "psa_progr"]]
        metadata = metadata.convert_dtypes()
        metadata = metadata.astype("category")

        annotations = data["annotations"]
        data = compute_label_frequency(annotations, level="label", log_scale=True)
        data = data.reset_index().pivot(index="sample_id", columns="label", values="proportion")
        data.columns = data.columns.astype(str)  # note: pyarrow cannot handle categorical columns
        data = data.astype(float)

        data, metadata = data.align(metadata, axis=0)
        assert data.isna().any().any() == False
        assert metadata.isna().any().any() == False

        if not self.data_path.exists():
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            data.to_parquet(self.data_path)

        if not self.metadata_path.exists():
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata.to_parquet(self.metadata_path)
