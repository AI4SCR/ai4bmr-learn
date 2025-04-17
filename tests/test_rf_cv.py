from ai4bmr_learn.data_models import WandbInitConfig
from ai4bmr_learn.datamodules.DummyTabular import DummyTabular
from ai4bmr_learn.routines.rf_cv import rf_cv, SweepConfig


# %% DATA
dm = datamodule = DummyTabular()
dm.prepare_data()
dm.setup()

# %%
wandb_init = WandbInitConfig(project='default')
sweep = SweepConfig()
rf_cv(datamodule=dm, sweep=sweep, wandb_init=wandb_init)