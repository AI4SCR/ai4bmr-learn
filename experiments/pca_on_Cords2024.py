# %%
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

from ai4bmr_learn.plotting.utils import get_grid_dims
import torch

# HELPER
def normalize(img, censoring=0.99, cofactor=1, exclude_zeros=True):
    img = np.arcsinh(img / cofactor)

    if exclude_zeros:
        masked_img = np.where(img == 0, np.nan, img)
        thres = np.nanquantile(masked_img, censoring, axis=(1, 2), keepdims=True)
    else:
        thres = np.quantile(img, q=censoring, axis=(1, 2), keepdims=True)

    img = np.minimum(img, thres)

    return img

# %% DATASET
from ai4bmr_datasets import Cords2024
base_dir = Path("/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/Cords2024")
dm = Cords2024(base_dir=base_dir)
dm.setup(image_version='published', mask_version='published')
images = dm.images

global_pca = PCA(n_components=3, random_state=0)
num_images = 100
rng = np.random.default_rng(0)
sample_ids = rng.choice(list(images.keys()), size=num_images, replace=False)

# %% PCA
stack = []
for i, img_id in enumerate(sample_ids[:100], start=1):
    logger.info(f"Processing PCA for image {i}/{num_images}: {img_id}")
    img = images[img_id].data

    img = normalize(img)
    img_flat = img.reshape(img.shape[0], -1).T
    stack.append(img_flat)

stack = np.vstack(stack)
global_pca.fit(stack)

# %%
def normalize_image(image: np.ndarray) -> np.ndarray:
    image_min = image.min()
    image_max = image.max()
    if image_max == image_min:
        return np.zeros_like(image, dtype=np.float32)  # Avoid division by zero
    return (image - image_min) / (image_max - image_min)

sample_wise_pca = True
rng = np.random.default_rng(0)
sample_ids = rng.choice(list(images.keys()), size=16, replace=False)
nrows, ncols = get_grid_dims(len(sample_ids))
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=(ncols * 3, nrows * 3))
grid = []
for img_id, ax in zip(sample_ids, axs.flat):
    img = images[img_id].data

    img = normalize(img)

    c, h, w = img.shape

    if h < 256 or w < 256:
        logger.warning(f"Ignoring small image: {img_id} with {h}x{w}")
        continue

    img = normalize(img)

    if sample_wise_pca:
        pca = PCA(n_components=3, random_state=0)
        img_flat = img.reshape(c, -1).T
        img = pca.fit_transform(img_flat).T.reshape(3, h, w)
    else:
        img_flat = img.reshape(c, -1).T
        img = global_pca.transform(img_flat).T.reshape(3, h, w)

    img = normalize_image(img)

    img_t = torch.tensor(img)
    grid.append(img_t)
    ax.imshow(img_t.permute(1,2,0))
    # ax.set_axis('off')

fig.tight_layout()
fig.show()

# %%
# from torchvision.utils import make_grid
# grid = make_grid(grid, nrow=ncols, padding=2, normalize=True, scale_each=False)
# plt.imshow(grid.permute(1, 2, 0)).figure.show()
