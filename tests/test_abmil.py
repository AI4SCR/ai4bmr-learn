from ai4bmr_learn.datamodules.DummyMIL import DummyMIL
from ai4bmr_learn.routines.abmil import abmil
from ai4bmr_learn.train.train import TrainerConfig

trainer = TrainerConfig(max_epochs=2, log_every_n_steps=1)
dm = DummyMIL()
abmil(datamodule=dm, trainer=trainer)


# %%
from ai4bmr_learn.models.mil.ABMIL import ABMILModule
import torchinfo

module = ABMILModule(
    num_classes=2,
    feature_dim=1024,
    class_weight=None,
    head_dim=4,
    n_heads=1,
    dropout=0.0,
    hidden_dim=4,
    pre_attention=False,
    pre_attention_dim=None,
    post_attention=False,
)
torchinfo.summary(module, verbose=0)

module = ABMILModule(
    num_classes=2,
    feature_dim=1024,
    class_weight=None,
    head_dim=2,
    n_heads=1,
    # dropout=0.0,
    hidden_dim=2,
    pre_attention_dim=None,
    post_attention=False,
)
torchinfo.summary(module, verbose=0)
