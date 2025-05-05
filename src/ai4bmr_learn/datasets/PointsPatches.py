from torch.utils.data import Dataset
from dataclasses import asdict
from ai4bmr_learn.data_models.Coordinate import PointsCoordinate
from ai4bmr_learn.utils.utils import pair
from ai4bmr_learn.utils.images import get_points
import openslide
import torch

class PointsPatches(Dataset):
    def __init__(self, coords: list[PointsCoordinate], transform=None,
                 metadata_keys=['object_id', 'label']):
        super().__init__()

        self.coords = coords
        self.transform = transform
        self.metadata_keys = metadata_keys

    def item_to_data(self, item):
        points = item.pop('points')
        data_names = sorted((set(points.columns) - set(self.metadata_keys) - {'geometry'}))
        data = points[data_names].values

        x = points.geometry.x.values
        y = points.geometry.y.values

        metadata = {}
        metadata['points'] = points[self.metadata_keys].to_dict(orient='list')
        metadata['coord'] = item

        item = dict(
            data=torch.tensor(data).float(),
            x=x,
            y=y,
            metadata=metadata,
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

        # TODO: find a better solution that this
        if 'views' in item:
            views = [self.item_to_data(view) for view in item['views']]
            item = self.item_to_data(item['item'])
            data = dict(views=views, item=item)
        else:
            data = self.item_to_data(item)

        return data