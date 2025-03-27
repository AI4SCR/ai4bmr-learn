from ai4bmr_learn.datamodules.DummyTabular import DummyDataModule

dm = DummyDataModule()
dm.prepare_data()
dm.setup()