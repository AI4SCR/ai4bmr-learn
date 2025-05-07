from torch.utils.data import Dataset
from dataclasses import asdict
from ai4bmr_learn.data_models.Coordinate import PointsCoordinate
from ai4bmr_learn.transforms.multiview import MultiViewTransform
from ai4bmr_learn.utils.utils import pair
from ai4bmr_learn.utils.images import get_points
import openslide
import torch

class PointsTokenizer:

    def __init__(self, labels: list[str]):
        self.labels = labels
        self.label_to_id = {label: i for i, label in enumerate(labels)}
        self.id_to_label = {i: label for i, label in enumerate(labels)}

    def __call__(self, points):
        pass

class PointsPatches(Dataset):
    def __init__(self, coords: list[PointsCoordinate], labels: list[str], label_key: str = 'label', transform=None):
        super().__init__()

        self.coords = coords
        self.labels = labels

        self.label_key = label_key
        self.transform = transform

        self.is_multiview = isinstance(self.transform, MultiViewTransform)

    def item_to_data(self, item):
        points = item.pop('points')
        # data_names = sorted((set(points.columns) - set(self.metadata_keys) - {'geometry'}))

        counts = points[self.label_key].value_counts()
        counts = counts.reindex(self.labels).fillna(0).values

        # x = points.geometry.x.values
        # y = points.geometry.y.values
        #
        # metadata = {}
        # metadata['points'] = points[self.metadata_keys].to_dict(orient='list')
        # metadata['coord'] = item
        #
        item = dict(
            data=torch.tensor(counts).float(),
        #     x=x,
        #     y=y,
        #     metadata=metadata,
        )

        return item

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]

        points = get_points(coord)

        item = {**asdict(coord)}
        item["points"] = points

        if self.transform:
            item = self.transform(item)

        if self.is_multiview:
            views = [self.item_to_data(view) for view in item['views']]
            item = self.item_to_data(item['item'])
            data = dict(views=views, item=item)
        else:
            data = self.item_to_data(item)

        return data