from ai4bmr_learn.datamodules.dataloader_collection import DatasetLoaderCollection, DataloaderConfig

from torchvision.datasets import CIFAR10
from torchvision.transforms import v2


base_dir = '/users/amarti51/prometex/data/datasets/cifar10'
transform = v2.ToImage()
ds = CIFAR10(root=base_dir, train=True, download=True, transform=transform)
dl = DataloaderConfig()

collection = DatasetLoaderCollection(
    datasets=dict(fit=[ds]),
    dataloaders=dict(fit=[dl]))
collection.setup()
dl_train = collection.train_dataloader()
batch = next(iter(dl_train))
