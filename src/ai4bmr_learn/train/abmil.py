# %%
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import wandb
from loguru import logger
from lightning import seed_everything
from torchmetrics import MetricCollection
from ai4bmr_learn.data_models.WandInitConfig import WandbInitConfig
from ai4bmr_learn.datamodules.MIL import MILDataModule
from ai4bmr_learn.models.mil.ABMIL import ABMILModule
from ai4bmr_learn.train.train import TrainerConfig, get_trainer

from ..models.utils import collect_model_stats
from ..logging.class_distribution import log_class_distribution
from ..logging.confusion_matrix import log_confusion_matrix
from ..metrics.classification import get_metric_collection

# from ai4bmr_learn.metrics.classification import get_metric_collection
# from ai4bmr_learn.models.utils import collect_model_stats


@dataclass
class ABMILConfig:
    head_dim: int = 256
    n_heads: int = 1
    dropout: float = 0.0
    n_branches: int = 1
    gated: bool = False
    hidden_dim: int = 256


# %%
def setup_wandb(wandb_init: WandbInitConfig, config=None):
    if wandb.run is not None:
        logger.info(f"Active wandb run found ({wandb.run.name}). Skipping wandb init.")
        return True
    else:
        wandb.init(**asdict(wandb_init), config=config)
        return False


# %%
def abmil(
    datamodule: MILDataModule,
    weighted: bool = False,
    monitor_metric_name: str = "val_loss_epoch",
    model: ABMILConfig = ABMILConfig(),
    trainer: TrainerConfig = TrainerConfig(),
    wandb_init: WandbInitConfig = WandbInitConfig(),
    metric_collection: MetricCollection = None,
    viz_collection: tuple | None = ("confusion_matrix", "class_distribution"),
    metadata: dict = None,
    metric_identifier: str = "scores",
):

    metadata = metadata or {}

    # DATA
    dm = datamodule
    dm.prepare_data()
    dm.setup()

    labels = dm.dataset.labels

    if weighted:
        # TODO: should we use the test set here for the class distribution?
        class_distribution = dm.dataset.class_distribution
        class_weight = 1 / class_distribution
        class_weight /= class_weight.sum()
    else:
        class_weight = None

    # METRICS
    metric_collection = metric_collection or get_metric_collection(num_classes=dm.dataset.num_classes)
    metrics_train = metric_collection.clone(prefix=f"{metric_identifier}:train/")
    metrics_test = metrics_train.clone(prefix=f"{metric_identifier}:test/")

    # MODEL
    module = ABMILModule(
        num_classes=dm.dataset.num_classes,
        feature_dim=dm.dataset.num_features,
        class_weight=class_weight,
        **asdict(model),
    )
    model_stats = collect_model_stats(module)

    # RUN CONFIGURATION
    config = {
        **asdict(model),
        **model_stats,
        **asdict(trainer),
        "weighted": weighted,
        "target_column_name": dm.target_column_name,
    }
    has_active_run = setup_wandb(wandb_init=wandb_init, config=config)

    # METADATA
    ckpt_dir = Path(wandb.run.dir) / "checkpoints" / wandb_init.project / wandb.run.name

    # TRAIN
    trainer = get_trainer(config=trainer, monitor_metric_name=monitor_metric_name, ckpt_dir=ckpt_dir)
    seed_everything(42, workers=True)
    trainer.fit(model=module, datamodule=dm)

    # BEST MODEL
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = ABMILModule.load_from_checkpoint(str(best_model_path))

    # PREDICTIONS
    test_results = trainer.predict(model=best_model, dataloaders=dm.test_dataloader())
    y_test_pred = torch.concat([i["prediction"] for i in test_results])
    y_test = torch.concat([i["target"] for i in test_results])
    # TODO: remove after testing
    assert (dm.dataset.metadata.set_index("split").loc["test"].values.flatten() == y_test.numpy()).all()

    train_results = trainer.predict(model=best_model, dataloaders=dm.train_dataloader())
    y_train_pred = torch.concat([i["prediction"] for i in train_results])
    y_train = torch.concat([i["target"] for i in train_results])

    # TEST SCORES
    test_scores = metrics_test(y_test_pred, y_test)
    wandb.log({**test_scores, **metadata})

    # TRAIN SCORES
    train_scores = metrics_train(y_train_pred, y_train)
    wandb.log({**train_scores, **metadata})

    # VISUALIZATION
    records = [
        dict(split="train", y_true=y_train, y_pred=y_train_pred, labels=labels),
        dict(split="test", y_true=y_test, y_pred=y_test_pred, labels=labels),
    ]

    if (viz_collection is not None) and ("class_distribution" in viz_collection):
        log_class_distribution(records=records, metadata=metadata)
    if (viz_collection is not None) and ("confusion_matrix" in viz_collection):
        log_confusion_matrix(records=records, metadata=metadata)

    if not has_active_run:
        wandb.config.update({"ckpt_dir": str(ckpt_dir)})
        wandb.config.update({"best_model_path": str(best_model_path)})
        wandb.finish()

    train_scores = {k: v.item() for k, v in train_scores.items()}
    test_scores = {k: v.item() for k, v in test_scores.items()}

    return train_scores, test_scores, model_stats
