from torch.utils.data._utils.collate import collate, default_collate_fn_map
import torch

class Concat:

    def __init__(self):
        pass

    def concat(self, batch: list):

        if isinstance(batch[0], torch.Tensor):
            return torch.cat(batch)

        if isinstance(batch[0], (int, float)):
            return torch.tensor(batch)

        if isinstance(batch[0], (str)):
            return list(batch)

        if isinstance(batch[0], (list, tuple)):
            new = []
            for item in batch:
                new.extend(self.concat(item))
            return new

        if isinstance(batch[0], dict):
            new = {}
            keys = list(batch[0].keys())
            for key in keys:
                new[key] = self.concat([i[key] for i in batch])
            return new

    def __call__(self, batch):
        return self.concat(batch)

