from ai4bmr_learn.datamodules.DummyTabular import DummyTabular

dm = DummyTabular()
dm.prepare_data()
dm.setup()

from ai4bmr_learn.datamodules.DummyMIL import DummyMIL
mil = DummyMIL()
