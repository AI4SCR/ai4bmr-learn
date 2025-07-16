from lightning.pytorch.callbacks import Callback
from loguru import logger

default_keys = ['backbone', 'head',  # clf, mae
                'tokenizer', 'encoder', 'decoder', 'proj', # mae
                'student_backbone', 'student_head', 'teacher_backbone', 'teacher_head'  # dino
                ]

class LogModelStats(Callback):

    def __init__(self, keys: list[str] = default_keys):
        self.keys = keys

    def on_fit_start(self, trainer, pl_module):
        from ai4bmr_learn.models.utils import collect_model_stats
        logger.info('Logging model statistics')

        stats = collect_model_stats(pl_module)
        for key in self.keys:
            if hasattr(pl_module, key):
                attr = getattr(pl_module, key)
                attr_stats = collect_model_stats(attr)
                attr_stats = {f'{key}.{k}':v for k, v in attr_stats.items()}
                stats.update(attr_stats)
        trainer.logger.experiment.config.update(stats)

