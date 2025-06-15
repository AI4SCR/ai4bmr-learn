from pathlib import Path
from ai4bmr_learn.datamodules.image_embedding import ImageEmbedding
from ai4bmr_learn.datasets.cifar10 import CIFAR10
from ai4bmr_learn.models.encoder.vision_encoder_torch_hub import VisionFeatureExtractor
from torchvision.transforms import v2
import torch

transform = v2.Compose([
    v2.Resize(224),
    v2.ToImage(),
    v2.ToDtype(dtype=torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
])
ds = CIFAR10(transform=transform)
backbone = VisionFeatureExtractor(model_name='resnet18')
dm = ImageEmbedding(data_path=Path('/users/amarti51/prometex/data/ai4bmr-learn/cifar10-reset18/data.parquet'),
                     metadata_path=Path('/users/amarti51/prometex/data/ai4bmr-learn/cifar10-resnet18/metadata.parquet'),
                     dataset=ds, backbone=backbone)
dm.prepare_data()
dm.setup()

# %%
dm.tabular.data.shape

# %%
import umap
import umap.plot

reducer = umap.UMAP()
reducer.fit(dm.tabular.data)

ax = umap.plot.points(reducer, labels=dm.tabular.targets, theme='fire')
ax.figure.show()

# %%