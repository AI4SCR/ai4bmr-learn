import numpy as np
import torch
import wandb
from glom import glom
from loguru import logger
from lightning.pytorch.callbacks import Callback

from ai4bmr_learn.callbacks.cache import ValidationCache


class ClassDistribution(Callback):
    """
    Logs a table (and bar chart) of the class distribution from ValidationCache outputs.

    Args:
        target_key: glom-style key to extract the discrete targets from each cached batch,
            e.g. "metadata.disease_state" or "y".
        every_num_epochs: run every N epochs (default 1 = every epoch).
        num_samples: optional cap on the number of targets to consider (after concatenation).
        class_names: optional mapping {class_id -> class_name} for prettier labels in W&B.
        table_prefix: W&B key prefix (namespace) for logs.
    """

    def __init__(
        self,
        target_key: str,
        num_samples: int | None = None,
        class_names: dict[int, str] | None = None,
        table_prefix: str = "class_distribution/",
        on_train: bool = True,
        on_val: bool = True,
    ):
        self.target_key = target_key
        self.num_samples = num_samples
        self.class_names = class_names or {}
        self.table_prefix = table_prefix

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        cache = self.get_validation_cache(trainer)
        if cache is None:
            logger.warning("ClassDistribution: no ValidationCache found; skipping.")
            return

        targets = self.collect_targets(cache)
        if targets.size == 0:
            logger.warning("ClassDistribution: collected 0 targets; skipping.")
            return

        if self.num_samples is not None:
            targets = targets[: self.num_samples]

        # compute distribution
        classes, counts = np.unique(targets, return_counts=True)
        total = counts.sum()
        pcts = counts / total

        # build W&B table
        rows = []
        for cls, cnt, pct in zip(classes, counts, pcts):
            cls_int = int(cls) if isinstance(cls, (np.integer, int)) else cls
            label = self.class_names.get(cls_int, str(cls_int))
            rows.append([label, int(cnt), float(pct)])

        table = wandb.Table(columns=["class", "count", "pct"], data=rows)

        logger.info(f"ClassDistribution [N={total}, epoch={trainer.current_epoch}]")
        if not trainer.fast_dev_run:
            exp = trainer.logger.experiment
            exp.log(
                {
                    f"{self.table_prefix}/table": table,
                }
            )

            try:
                chart = wandb.plot.bar(
                    table, "class", "count", title=f"{self.table_prefix} (val)"
                )
                exp.log({f"{self.table_prefix}/bar": chart})
            except Exception as e:
                logger.warning(f"ClassDistribution: failed to log bar chart: {e}")


    def get_validation_cache(self, trainer) -> ValidationCache | None:
        for cb in trainer.callbacks:
            if isinstance(cb, ValidationCache):
                return cb
        return None

    def collect_targets(self, cache) -> np.ndarray:
        """
        Pull targets from cache.outputs using glom(self.target_key),
        flatten to 1D numpy array of ints.
        """
        chunks: list[np.ndarray] = []

        for batch in cache.outputs:
            try:
                t = glom(batch, self.target_key)
            except Exception as e:
                logger.warning(f"ClassDistribution: glom failed for key '{self.target_key}': {e}")
                continue

            # normalize to numpy 1D
            if isinstance(t, torch.Tensor):
                arr = t.detach().cpu().numpy()
            elif isinstance(t, (list, tuple)):
                arr = np.asarray(t)
            elif isinstance(t, np.ndarray):
                arr = t
            else:
                # scalar -> wrap
                arr = np.asarray([t])

            if arr.ndim > 1:
                arr = arr.reshape(-1)
            chunks.append(arr)

        out = np.concatenate(chunks, axis=0)
        return out