import json
from pathlib import Path
import json
import numpy as np
import scipy
import torch
from torch.utils.data._utils.collate import collate, default_collate_fn_map

from ai4bmr_graphs.RadiusGraphBuilder import RadiusGraphBuilder
from ai4bmr_datasets.utils.imc.normalize import normalize

from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse, from_scipy_sparse_matrix

from sklearn.preprocessing import StandardScaler

import numpy as np
from skimage.measure import regionprops_table
import pandas as pd


# %%
def spatial_features(
    mask: np.ndarray,
    properties: tuple[str] = ("label", "centroid", "eccentricity", "area", "extent"),
):
    rp = regionprops_table(mask, properties=properties)
    feat = pd.DataFrame.from_dict(rp).set_index("label").rename(columns={"centroid-0": "y", "centroid-1": "x"})
    feat.index.name = "object_id"
    return feat


def intensity_features(
    img: np.ndarray,
    mask: np.ndarray,
    panel: pd.DataFrame | None = None,
):
    img = np.moveaxis(img, 0, -1)
    rp = regionprops_table(mask, img, properties=("label", "intensity_mean"))
    feat = pd.DataFrame.from_dict(rp).set_index("label")

    if panel is not None:
        mapping = {f"intensity_mean-{num}": panel.index[num] for num in range(len(panel))}
        feat = feat.rename(mapping, axis=1)

    feat.index.name = "object_id"
    return feat

class Graph:
    def __init__(self, radius: int, include_self: bool,
                 stats_path: Path, cache_dir: Path | None = None,
                 keep_mask: bool = False, keep_image: bool = False):

        self.radius = radius
        self.include_self = include_self
        self.keep_mask = keep_mask
        self.keep_image = keep_image
        self.builder = RadiusGraphBuilder()

        self.collate_fn_map = default_collate_fn_map.copy()
        self.collate_fn_map.update({Data: self.collate_graphs})

        self.cache_dir = None
        self.use_caching = False
        if cache_dir is not None:
            self.use_caching = True
            self.cache_dir = cache_dir.resolve()
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        assert stats_path.exists()
        with open(stats_path, 'r') as f:
            stats = json.load(f)

        # NOTE: instead of loading stats we could use self.scaler.partial_fit()
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array(stats['mean'])
        self.scaler.scale_ = np.array(stats['std'])
        self.scaler.n_features_in_ = len(stats['mean'])

    def collate_graphs(self, graphs: list[Data], collate_fn_map=None) -> dict[str, torch.Tensor]:
        return Batch.from_data_list(graphs)

    def process_batch(self, batch):
        new_batch = []

        for batch_item in batch:
            if 'views' in batch_item:
                views = [self.process_item(item) for item in batch_item['views']]
                batch_item['views'] = views
            else:
                batch_item = self.process_item(batch_item)

            new_batch.append(batch_item)

        return new_batch

    def process_item(self, item: dict) -> dict:
        if self.use_caching:
            cache_path = self.cache_dir / f'{item['sample_id']}.pt'

            if cache_path.exists():
                data = torch.load(cache_path, weights_only=False)
                item['graph'] = data
                if not self.keep_image:
                    item.pop('image').numpy()
                if not self.keep_mask:
                    item.pop('mask').numpy()
                return item

        image = item['image'].numpy() if self.keep_image else item.pop('image').numpy()
        mask = item['mask'].numpy() if self.keep_mask else item.pop('mask').numpy()

        result = self.builder.from_mask(mask=mask, radius=self.radius, include_self=self.include_self)
        adj = result.adj

        edge_index, edge_weight = from_scipy_sparse_matrix(adj)

        intensity = intensity_features(img=image, mask=mask)
        index = intensity.index
        intensity = normalize(intensity, scale=None, exclude_zeros=True)
        # intensity.boxplot().figure.show()
        intensity = self.scaler.transform(intensity.values)
        intensity = pd.DataFrame(intensity, index=index)
        # intensity.boxplot().figure.show()

        spatial = spatial_features(mask=mask)
        # spatial.boxplot().figure.show()
        index = spatial.index
        spatial = StandardScaler().fit_transform(spatial.values)
        spatial = pd.DataFrame(spatial, index=index)
        # spatial.boxplot().figure.show()

        x = pd.concat([intensity, spatial], axis=1)
        x = x.loc[result.object_ids]
        assert len(x) == len(intensity)
        # x.boxplot().figure.show()

        x = torch.tensor(x.values).float()
        data = Data(x=x, edge_index=edge_index)
        data.validate(raise_on_error=True)

        data.radius = self.radius
        data.include_self = self.include_self

        item['graph'] = data
        if self.use_caching:
            torch.save(data, cache_path)
        return item

    def collate(self, batch) -> dict:
        batch = self.process_batch(batch)
        result = collate(batch, collate_fn_map=self.collate_fn_map)  # pyright: ignore
        return result

    def __call__(self, batch):
        res = self.collate(batch)
        return res
