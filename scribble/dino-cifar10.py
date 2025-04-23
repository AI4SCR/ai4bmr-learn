# This example requires the following dependencies to be installed:
# pip install lightly

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import copy
from typing import Any

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule


class DINO(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        input_dim = 512
        # instead of a resnet you can also use a vision transformer backbone as in the
        # original paper (you might have to reduce the batch size in this case):
        # backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
        # input_dim = backbone.embed_dim

        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        views = batch[0]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        return loss

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return optim


model = DINO()

transform = DINOTransform()
# we ignore object detection annotations by setting target_transform to return 0


def target_transform(t):
    return 0


dataset = torchvision.datasets.VOCDetection(
    "/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/pascal_voc",
    download=True,
    transform=transform,
    target_transform=target_transform,
)

dataset = torchvision.datasets.CIFAR10(
    "/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/cifar10",
    download=True,
    transform=transform,
    target_transform=target_transform,
)

class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        img, target = super()[item]
        return {'image': img, 'target': target}
ds = CIFAR10(root="/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/cifar10")

# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

# %%
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
trainer = pl.Trainer(max_epochs=10, devices=1, accelerator=accelerator)

# %%
pre_train_embeddings = trainer.predict(model=model, dataloaders=dataloader)

# %%
trainer.fit(model=model, train_dataloaders=dataloader)

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
