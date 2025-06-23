import lightning as L
import torch
from ai4bmr_datasets.datasets.cifar10 import CIFAR10
from lightly.transforms.utils import IMAGENET_NORMALIZE
from matplotlib import pyplot as plt
from torchvision.transforms import v2
from torch.utils.data import Subset

from ai4bmr_learn.ssl.dino_light import DINOLight
from ai4bmr_learn.transforms.dino_transform import DINOTransformLightly

# %% SSL MODULE
model = DINOLight()

# %% TRANSFORMS
transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_NORMALIZE['mean'], std=IMAGENET_NORMALIZE['std'])
])

viz_transform = DINOTransformLightly(normalize=None)
dino_transform = DINOTransformLightly()

# %% DATA
random_indices = torch.randperm(50000)[:5000]
ds_viz = Subset(CIFAR10(transform=viz_transform), indices=random_indices)
ds_test = Subset(CIFAR10(transform=transform), indices=random_indices)
ds_train = Subset(CIFAR10(transform=dino_transform), indices=random_indices)

# %% VISUALIZE TRANSFORM
item = ds_viz[0]
fig, axs = plt.subplots(3, 3)
imgs = [item['item']['image']] + [i['image'] for i in item['views']]

for img, ax in zip(imgs, axs.flatten()):
    ax.imshow(img.permute(1, 2, 0))

fig.tight_layout()
fig.show()

# %%
dl_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

item = ds_train[0]
batch = next(iter(dl_train))

dl_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

# %%
trainer = L.Trainer(max_epochs=50, devices=1)

# %%
pre_train_embeddings = trainer.predict(model=model, dataloaders=dl_test)
pre_train_targets = torch.cat([emb['target'] for emb in pre_train_embeddings])
pre_train_embeddings = torch.cat([emb['embedding'] for emb in pre_train_embeddings])
pre_train_embeddings = pre_train_embeddings.squeeze()

# %%
trainer.fit(model=model, train_dataloaders=dl_train)

# %%
post_train_embeddings = trainer.predict(model=model, dataloaders=dl_test)
post_train_targets = torch.cat([emb['target'] for emb in post_train_embeddings])
post_train_embeddings = torch.cat([emb['embedding'] for emb in post_train_embeddings])
post_train_embeddings = post_train_embeddings.squeeze()

# %%
from ai4bmr_learn.plotting.umap import plot_umap

fig, axs = plt.subplots(1,2, figsize=(10, 5))
ax = plot_umap(pre_train_embeddings, labels=pre_train_targets, ax=axs[0])
ax.set_title('Pre-training')
ax = plot_umap(post_train_embeddings, labels=post_train_targets, ax=axs[1])
ax.set_title('Post-training')
ax.figure.show()
