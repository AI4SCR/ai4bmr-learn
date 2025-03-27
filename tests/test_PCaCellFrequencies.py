from ai4bmr_learn.datamodules.PCaCellFrequencies import PCaCellFrequenciesDataModule
from pathlib import Path

data_path = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/ai4bmr-learn/PCa/data.parquet')
metadata_path = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/ai4bmr-learn/PCa/metadata.parquet')

dm = PCaCellFrequenciesDataModule(
    data_path=data_path,
    metadata_path=metadata_path,
    target_column_name='cause_of_death'
)
dm.prepare_data()
dm.setup()
