import torch
from glom import glom
from lightning.pytorch.callbacks import Callback
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from ai4bmr_learn.utils.device import batch_to_device


class LinearProbing(Callback):

    def __init__(self,
                 target_key: str,
                 run_before_train: bool = True,
                 run_after_train: bool = True,
                 run_every_num_epochs: int = 10,
                 num_samples: int | None = None,
                 num_splits: int = 5,
                 test_size: float = 0.25,
                 random_state: int = 0,
                 ):

        self.run_before_train = run_before_train
        self.run_after_train = run_after_train
        self.run_every_num_epochs = run_every_num_epochs
        self.num_samples = num_samples
        self.target_key = target_key
        self.num_splits = num_splits
        self.test_size = test_size
        self.random_state = random_state

        self.scoring = {
            # "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_micro": "f1_micro",
            "f1_macro": "f1_macro",
            "precision_micro": "precision_micro",
            "precision_macro": "precision_macro",
            "recall_micro": "recall_micro",
            "recall_macro": "recall_macro",
        }

        self.embeddings = self.targets = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0, force: bool = False):

        if not self.should_run(trainer=trainer, force=force):
            return

        is_accumulated = self.accumulate(outputs)
        # note: trainer.is_last_batch is not always reset for some reason
        is_last_batch = len(trainer.val_dataloaders) - 1 == batch_idx

        if is_last_batch:
            logger.info(f'Linear probing [num_samples={len(self.embeddings)}, epoch={trainer.current_epoch}, batch_idx={batch_idx}]')
            self.run_evaluation(trainer=trainer)
            self.reset()

    def run_evaluation(self, trainer):
        cv = StratifiedShuffleSplit(n_splits=self.num_splits, test_size=self.test_size, random_state=self.random_state)
        clf = LogisticRegression(max_iter=1000)

        x = self.embeddings[:self.num_samples].numpy() if self.num_samples else self.embeddings.numpy()
        targets = self.targets[:self.num_samples] if self.num_samples else self.targets
        y = LabelEncoder().fit_transform(targets)
        scores = cross_validate(estimator=clf, X=x, y=y, cv=cv, scoring=self.scoring, n_jobs=-1)
        scores = {k: v.mean() for k,v in scores.items()}

        scores["epoch"] = trainer.current_epoch
        scores = {f'linear_probing/{k}': v for k,v in scores.items()}
        trainer.logger.experiment.log(scores)

    def accumulate(self, outputs) -> bool:
        accumulate = (self.num_samples is None) or (len(self.embeddings) < self.num_samples)

        if accumulate and self.embeddings is None:
            self.embeddings = outputs['embedding']
            targets = glom(outputs, self.target_key)
            targets = targets.tolist() if isinstance(targets, torch.Tensor) else targets
            self.targets = targets
        elif accumulate:
            self.embeddings = torch.vstack((self.embeddings, outputs['embedding']))
            targets = glom(outputs, self.target_key)
            targets = targets.tolist() if isinstance(targets, torch.Tensor) else targets
            self.targets.extend(targets)
        else:
            return True
        return False

    def get_outputs(self, trainer, pl_module):
        dl = trainer.val_dataloaders
        assert pl_module.device.type == 'cuda'
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dl), desc='collecting outputs'):
                batch = batch_to_device(batch, pl_module.device)
                outputs = pl_module.validation_step(batch=batch, batch_idx=batch_idx)
                is_accumulated = self.accumulate(outputs)

                if is_accumulated or trainer.fast_dev_run:
                    break

        return outputs, batch_idx

    def on_train_start(self, trainer, pl_module):
        if self.run_before_train:
            logger.info(f'Linear probing on_train_start')
            outputs, batch_idx = self.get_outputs(trainer, pl_module)
            self.accumulate(outputs)
            self.run_evaluation(trainer=trainer)
            self.reset()

    def on_train_end(self, trainer, pl_module):
        if self.run_after_train:
            logger.info(f'Linear probing on_train_end')
            outputs, batch_idx = self.get_outputs(trainer, pl_module)
            self.accumulate(outputs)
            self.run_evaluation(trainer=trainer)
            self.reset()

    def should_run(self, trainer, force):
        if trainer.sanity_checking:
            return False

        if force:
            return True

        if trainer.current_epoch == 0 and self.run_before_train:
            return False

        if trainer.current_epoch % self.run_every_num_epochs == 0:
            return True

        return False

    def reset(self):
        self.embeddings = self.targets = None

