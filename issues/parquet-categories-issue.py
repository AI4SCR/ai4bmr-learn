metadata.to_parquet(self.metadata_path, engine="fastparquet")
metadata.to_parquet(self.metadata_path, engine="pyarrow")

tmp = pd.read_parquet(self.metadata_path, engine="fastparquet")
tmp.cause_of_death.dtype
tmp = tmp.convert_dtypes()
tmp.cause_of_death.dtype
pd.read_parquet(self.metadata_path, engine="fastparquet").cause_of_death.dtype
pd.read_parquet(self.metadata_path, engine="fastparquet").os_status.dtype
pd.read_parquet(self.metadata_path, engine="fastparquet").psa_progr.dtype

from lightning.pytorch.callbacks import ModelCheckpoint