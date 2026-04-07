from __future__ import annotations

from collections.abc import Sequence

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from loguru import logger

from ai4bmr_learn.models.utils import collect_model_stats

DEFAULT_KEYS = (
    "backbone",
    "mil",
    "head",
    "tokenizer",
    "encoder",
    "decoder",
    "proj",
    "student_backbone",
    "student_head",
    "teacher_backbone",
    "teacher_head",
)


class LogModelStats(Callback):
    def __init__(self, keys: Sequence[str] = DEFAULT_KEYS) -> None:
        super().__init__()
        self.keys = tuple(keys)

    def on_fit_start(self, trainer: Trainer, pl_module) -> None:
        if trainer.fast_dev_run:
            return

        logger_ = trainer.logger
        if logger_ is None or not hasattr(logger_, "experiment"):
            return

        logger.info("Logging model statistics")

        stats = collect_model_stats(pl_module)
        for key in self.keys:
            if not hasattr(pl_module, key):
                continue
            prefixed_stats = {
                f"{key}.{name}": value for name, value in collect_model_stats(getattr(pl_module, key)).items()
            }
            stats.update(prefixed_stats)

        logger_.experiment.config.update(stats)
