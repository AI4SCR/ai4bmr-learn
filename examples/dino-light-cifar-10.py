import lightning as L
import torch
import torchvision
from ai4bmr_datasets.datasets.CIFAR10 import CIFAR10

from ai4bmr_learn.ssl.dino_light import DINOLight
from ai4bmr_learn.transforms.dino_transform import DINOTransform

# %% SSL MODULE
model = DINOLight()

# %% DATA
transform = DINOTransform(normalize=None)
# ds = CIFAR10()
ds = CIFAR10(transform=transform)
item = ds[0]

from matplotlib import pyplot as plt
fig, axs = plt.subplots(3, 3)
imgs = [item['image']] + item['global_views'] + item['local_views']
for img, ax in zip(imgs, axs.flatten()):
    ax.imshow(img.permute(1,2,0))

fig.tight_layout()
fig.show()

item.keys()


dl = torch.utils.data.DataLoader(
    ds,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

# %%
trainer = L.Trainer(max_epochs=10, devices=1)

# %%
pre_train_embeddings = trainer.predict(model=model, dataloaders=dl)

# %%
trainer.fit(model=model, train_dataloaders=dl)

# %%
from torchvision.transforms import v2
from lightly.transforms.utils import IMAGENET_NORMALIZE
transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_NORMALIZE['mean'], std=IMAGENET_NORMALIZE['std'])
])
dataset = torchvision.datasets.CIFAR10(
    "/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/cifar10",
    download=True,
    transform=transform,
    # target_transform=target_transform,
)
# inp, target = dataset[0]
# inp = inp.unsqueeze(0)

# %%
from tqdm import tqdm
student = model.student_backbone
student.eval()
student.to('cuda')

embeddings = []
targets = []
with torch.no_grad():
    for inp, target in tqdm(dataset):
        inp = inp.unsqueeze(0)
        inp = inp.to('cuda')
        out = student(inp)
        out = out.squeeze()
        embeddings.append(out.cpu())
        targets.append(target)

embeddings = torch.stack(embeddings)
targets = torch.tensor(targets)

# %%
from umap import UMAP
import umap.plot
reducer = UMAP(n_components=2)
reducer.fit(embeddings)

ax = umap.plot.points(reducer, labels=targets)
ax.figure.show()

# %%

# %%
import time
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.v2 as v2

# Create dummy image (e.g., 512x512 RGB)
dummy_array = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
patch = Image.fromarray(dummy_array)

# Target size
patch_width, patch_height = 256, 256


num_iters = 100

# --- PIL Resize ---
start_pil = time.time()
for _ in range(num_iters):
    resized_pil = patch.resize((patch_width, patch_height))
    arr1 = torch.tensor(np.asarray(patch)).permute(2, 0, 1) / 255
end_pil = time.time()
print(f"PIL resize time (100 runs): {end_pil - start_pil:.4f} seconds")

# --- torchvision v2 Resize ---
transform_v2 = v2.Compose([
    v2.ToImage(),  # assumes input is PIL or ndarray
    v2.Resize((patch_height, patch_width)),
    v2.ToDtype(torch.float32, scale=True),
])

start_v2 = time.time()
for _ in range(num_iters):
    arr2 = transform_v2(patch)
end_v2 = time.time()
print(f"torchvision v2 resize time (100 runs): {end_v2 - start_v2:.4f} seconds")


transform_v3 = v2.Resize((patch_height, patch_width))

start_v3 = time.time()
for _ in range(num_iters):
    arr3 = torch.tensor(np.asarray(patch)).permute(2, 0, 1)
    arr3 = transform_v3(arr3)
    arr3 = arr3 / 255
end_v3 = time.time()
print(f"torchvision v3 array resize time (100 runs): {end_v3 - start_v3:.4f} seconds")

# --- torchvision v4 Resize ---
transform_v4 = v2.Compose([
    v2.Resize((patch_height, patch_width)),
    v2.ToImage(),  # assumes input is PIL or ndarray
    v2.ToDtype(torch.float32, scale=True),
])

start_v4 = time.time()
for _ in range(num_iters):
    arr4 = transform_v4(patch)
end_v4 = time.time()
print(f"torchvision v4 resize time (100 runs): {end_v4 - start_v4:.4f} seconds")
