from ai4bmr_learn.routines.abmil import abmil
from ai4bmr_learn.datamodules.DummyMIL import DummyMIL
from ai4bmr_learn.train.train import TrainerConfig
from ai4bmr_learn.datamodules.NestedCV import NestedCV

trainer = TrainerConfig(max_epochs=2, log_every_n_steps=1)
dm = DummyMIL()
dm.prepare_data()
dm.setup()
save_dir = dm.splits_path.parent / "nested_cv_splits"
ncv = NestedCV(metadata=dm.dataset.metadata, save_dir=save_dir, test_size=0.2)
for outer_fold in range(ncv.num_outer_cv): break
    outer_split_path = ncv.get_dataset_path(outer_fold=outer_fold)
    for inner_fold in range(ncv.num_inner_cv): break
        inner_split_path = ncv.get_dataset_path(outer_fold=outer_fold, inner_fold=inner_fold)
        dm.splits_path = inner_split_path
        abmil(datamodule=dm, trainer=trainer)
