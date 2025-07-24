import numpy as np
import torch
from glom import glom
from lightning.pytorch.callbacks import Callback
from loguru import logger
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate

from ai4bmr_learn.callbacks.cache import ValidationCache

from sklearn.linear_model import Ridge, Lasso, ElasticNet


class LinearRegressionProbing(Callback):

    def __init__(self,
                 target_key: str,
                 before_train: bool = False,
                 after_train: bool = False,
                 every_num_epochs: int = 1,
                 num_samples: int | None = None,
                 num_splits: int = 5,
                 test_size: float = 0.25,
                 random_state: int = 0,
                 ):

        self.before_train = before_train
        self.after_train = after_train
        self.every_num_epochs = every_num_epochs

        self.num_samples = num_samples
        self.target_key = target_key

        self.num_splits = num_splits
        self.test_size = test_size
        self.random_state = random_state
        self.scoring = {
            "neg_mean_absolute_error": "neg_mean_absolute_error",
        }

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return

        logger.info(f'Run linear regression probing...')

        cache, = list(filter(lambda x: isinstance(x, ValidationCache), trainer.callbacks))
        x, targets = self.get_data(cache)
        self.run_evaluation(x, targets=targets, trainer=trainer)

    def get_data(self, cache) -> tuple[np.ndarray, np.ndarray]:
        x = torch.vstack([i['embedding'] for i in cache.outputs]).detach().cpu().numpy()
        targets = []
        for batch in cache.outputs:
            val = glom(batch, self.target_key)
            val = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else val
            targets.append(val)
        targets = np.concat(targets)

        x = x[:self.num_samples] if self.num_samples else x
        targets = targets[:self.num_samples] if self.num_samples else targets
        return x, targets


    def run_evaluation(self, x, targets, trainer):

        cv = StratifiedShuffleSplit(n_splits=self.num_splits, test_size=self.test_size, random_state=self.random_state)
        model = Ridge(alpha=1.0)  # L2-penalized
        # model = Lasso(alpha=0.1)  # L1-penalized
        # model = ElasticNet(alpha=0.1, l1_ratio=0.5)

        try:
            scores = cross_validate(estimator=model, X=x, y=targets, cv=cv, scoring=self.scoring, n_jobs=-1)
        except ValueError as e:
            logger.error(e)
            return

        scores = {k: v.mean() for k,v in scores.items()}

        scores = {f'linear_regression_probing/{k}': v for k,v in scores.items()}
        scores["epoch"] = trainer.current_epoch

        if not trainer.fast_dev_run:
            trainer.logger.experiment.log(scores)