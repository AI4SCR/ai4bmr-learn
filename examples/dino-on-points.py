# %%
from ai4bmr_datasets.datasets.DummyPoints import DummyPoints
from ai4bmr_learn.
from torch.utils.data import DataLoader

# %%
ds = DummyPoints()
dl = DataLoader(ds, batch_size=1)

# %%