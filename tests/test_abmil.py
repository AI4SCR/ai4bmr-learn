from ai4bmr_learn.routines.abmil import abmil
from ai4bmr_learn.datamodules.DummyMIL import DummyMIL
from ai4bmr_learn.train.train import TrainerConfig

trainer = TrainerConfig(max_epochs=2, log_every_n_steps=1)
dm = DummyMIL()
abmil(datamodule=dm, trainer=trainer)
