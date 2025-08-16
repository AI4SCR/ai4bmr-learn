import torch
from lightning import Trainer
from torch.utils.data import DataLoader

from ai4bmr_learn.models.classical.logistic_regression import LogisticRegression
from ai4bmr_learn.callbacks.cache import TrainCache, ValidationCache


X_train = torch.randn(10, 4)
y_train = torch.randint(0, 2, (10,))
train_ds = [{"x": X_train[i], "y": y_train[i]} for i in range(10)]

X_val = torch.randn(5, 4)
y_val = torch.randint(0, 2, (5,))
val_ds = [{"x": X_val[i], "y": y_val[i]} for i in range(5)]

train_loader = DataLoader(train_ds, batch_size=5, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=5, shuffle=False)

# --- LightningModule ---
model = LogisticRegression(
    batch_key="x",
    target_key="y",
    num_classes=2,
)

# cache callbacks
train_cache = TrainCache(num_samples=None)
val_cache = ValidationCache(num_samples=None)

trainer = Trainer(
    max_epochs=1,
    enable_checkpointing=False,
    enable_model_summary=False,
    log_every_n_steps=1,
    callbacks=[train_cache, val_cache],
    # default_root_dir=tmp_path
)

# %%
# train
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# 1) make sure all training samples were cached
assert len(train_cache.outputs) == len(train_loader)

# 2) check that the model can predict on validation X
preds = model.model.predict(X_val.numpy())
assert preds.shape[0] == len(val_ds)


# %%
import torch
from lightning import Trainer
from torch.utils.data import DataLoader

from ai4bmr_learn.models.classical.logistic_regression import LogisticRegression
from ai4bmr_learn.callbacks.cache import TrainCache, ValidationCache
from ai4bmr_learn.datasets.embeddings import Embeddings
from ai4bmr_learn.collators.concat import Concat
from pathlib import Path
ds = Embeddings(data_dir=Path('/users/amarti51/prometex/data/embeddings/geneformer/patch-level/test-0'))
ds.setup()
item = ds[0]

# if embeddings are batched I recommend to load them 1-by-1
collate_fn = Concat()
# batch = collate_fn.concat([item, item])
train_loader = DataLoader(ds, batch_size = 2, collate_fn = collate_fn)
val_loader = DataLoader(ds, batch_size = 1, collate_fn = collate_fn)
batch = next(iter(train_loader))

model = LogisticRegression(
    batch_key="z",
    target_key="metadata.disease_state",
    num_classes=3,
    pooling='flatten'
)

train_cache = TrainCache(num_samples=None)
trainer = Trainer(
    max_epochs=1,
    enable_checkpointing=False,
    enable_model_summary=False,
    log_every_n_steps=1,
    callbacks=[train_cache],
    # default_root_dir=tmp_path
)

# %%
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
