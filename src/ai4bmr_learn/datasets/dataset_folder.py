from pathlib import Path
from torch.utils.data import Dataset
import tifffile
import torch
from torchvision import tv_tensors
import pandas as pd
from ai4bmr_learn.data.splits import Split


class DatasetFolder(Dataset):

    def __init__(self,
                 dataset_dir: Path,
                 split_version: str,
                 split: str | None = None,
                 transform=None,
                 image_version: str | None = None,
                 target_name: str | None = None,
                 mask_version: str | None = None,
                 graph_version: str | None = None,
                 annotation_version: str | None = None,
                 ):
        super().__init__()

        self.dataset_dir = dataset_dir.resolve()
        assert dataset_dir.exists() and dataset_dir.is_dir()

        self.split_path = self.dataset_dir / 'splits' / f"{split_version}.parquet"
        self.split = split
        self.metadata = pd.read_parquet(self.split_path)

        if split is not None:
            filter_ = self.metadata[Split.COLUMN_NAME.value] == split
            self.sample_ids = self.metadata[filter_].index.tolist()
        else:
            self.sample_ids = self.metadata.index.tolist()

        self.target_name = target_name
        self.targets = self.metadata[target_name] if target_name else None

        self.image_version = image_version
        if image_version is not None:
            self.images_dir = self.dataset_dir / "images" / image_version
            assert self.images_dir.exists() and self.images_dir.is_dir()
            self.image_paths = {i.stem: i for i in self.images_dir.glob("*.tiff")}
            assert set(self.sample_ids) <= set(self.image_paths)

        self.mask_version = mask_version
        if mask_version is not None:
            self.masks_dir = self.dataset_dir / "masks" / mask_version
            assert self.masks_dir.exists() and self.masks_dir.is_dir()
            self.mask_paths = {i.stem: i for i in self.masks_dir.glob("*.tiff")}
            assert set(self.sample_ids) <= set(self.mask_paths)

        self.graph_version = graph_version
        if graph_version is not None:
            self.graphs_dir = self.dataset_dir / "graphs" / graph_version
            assert self.graphs_dir.exists() and self.graphs_dir.is_dir()
            self.graph_paths = {i.stem: i for i in self.graphs_dir.glob("*.pt")}
            assert set(self.sample_ids) <= set(self.graph_paths)

        self.annotation_version = annotation_version
        if annotation_version is not None:
            self.annotations_dir = self.dataset_dir / "annotations" / annotation_version
            assert self.annotations_dir.exists() and self.annotations_dir.is_dir()
            self.annotation_paths = {i.stem: i for i in self.annotations_dir.glob('*.tiff')}
            # assert set(self.sample_ids) <= set(self.annotation_paths)

        self.to_image = lambda x: tv_tensors.Image(torch.tensor(x).float())
        self.to_mask = lambda x: tv_tensors.Mask(torch.tensor(x).long())
        self.transform = transform

    def __getitem__(self, key):
        if isinstance(key, tuple):
            idx, sample_id = key
        else:
            sample_id = self.sample_ids[key]

        item = {'sample_id': sample_id}

        if self.image_version is not None:
            image_path = self.image_paths[sample_id]
            image = tifffile.imread(image_path)
            image = self.to_image(image)
            item['image'] = image

        if self.mask_version is not None:
            mask_path = self.mask_paths[sample_id]
            mask = tifffile.imread(mask_path)
            mask = self.to_mask(mask)
            item['mask'] = mask

        if self.graph_version is not None:
            graph_path = self.graph_paths[sample_id]
            graph = torch.load(graph_path, weights_only=False)
            item['graph'] = graph

        if self.targets is not None:
            target = self.targets[sample_id]
            item['metadata'] = {self.target_name: target}
            item['target'] = target

        if self.annotation_version is not None:
            if sample_id in self.annotation_paths:
                anno_path = self.annotation_paths[sample_id]
                anno = tifffile.imread(anno_path)
                keep = True
            else:
                anno = torch.zeros_like(image)
                keep = False

            anno = self.to_mask(anno)
            item['annotations'] = {self.annotation_version: {'annotation': anno, 'keep': keep}}

        if self.transform is not None:
            item = self.transform(item)

        return item

    def __len__(self):
        return len(self.sample_ids)

    def setup(self):
        pass
