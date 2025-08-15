from ai4bmr_learn.datamodules.dataloader_collection import DataLoaderCollection, DataLoader
from torchvision.datasets import CIFAR10


base_dir = '/users/amarti51/prometex/data/datasets/cifar10'
ds = CIFAR10(root=base_dir, train=True, download=True)

dl = DataLoader(ds)
collection = DataLoaderCollection(dataloaders=dict(fit=[dl]))
collection.setup()
dl_train = collection.train_dataloader()
batch = next(iter(dl_train))
