import torch
from ai4bmr_learn.datasets.mil import MILFromDataset
from torch.utils.data import Dataset, DataLoader
import tempfile
from pathlib import Path

def test_mil():
    class DummyDataset(Dataset):

        def __init__(self):
            self.bag_ids = torch.randint(0,3, size=(100,)).tolist()

        def __getitem__(self, idx):
            return dict(image=torch.randn(3, 224, 224), target=0 if torch.randn(1).item() > 0.5 else 0)

        def __len__(self):
            return len(self.bag_ids)

    ds = DummyDataset()

    mil = MILFromDataset(dataset=ds)
    mil.setup()
    bag = mil[0]

    mil = MILFromDataset(dataset=ds, num_instances=10)
    mil.setup()
    bag = mil[0]
    assert bag['image'].shape[0] == 10

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mil = MILFromDataset(dataset=ds, num_instances=10, cache_path=tmpdir)
        # mil.invalidate_cache()
        mil.setup()

        bag = mil[0]

        bag_id = mil.bag_ids[0]
        assert (tmpdir / f'{bag_id}.pkl').exists(), 'cached bag does not exist'

        bag = mil[0]
        assert bag['image'].shape[0] == 10

    dl = DataLoader([bag, bag], batch_size=2)
    batch = next(iter(dl))

    # %%
    from ai4bmr_learn.supervised.mil import MIL
    from ai4bmr_learn.models.backbones.timm import Backbone
    backbone = Backbone(model_name='resnet18', global_pool='avg')

    model = MIL(backbone=backbone)