from dataclasses import asdict, dataclass
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger


@dataclass
class TrainerConfig:
    max_epochs: int = 100
    min_epochs: int = 1
    accumulate_grad_batches: int = 1
    log_every_n_steps: int = 50
    precision: str | int = "32-true"
    accelerator: str = "auto"
    gradient_clip_val: float | None = None
    devices: int = 1
    fast_dev_run: bool = False


def get_trainer(
    config: TrainerConfig, ckpt_dir: Path, monitor_metric_name="val_loss_epoch", metadata: dict | None = None
):

    metadata = metadata or {}

    # LOGGER
    wandb_logger = WandbLogger()

    # CALLBACKS
    fname = f"{{epoch}}-{{{monitor_metric_name}:.4f}}"

    model_ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor=monitor_metric_name,
        mode="min",
        save_top_k=3,
        filename=fname,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    # early_stop = EarlyStopping(monitor=monitor_metric_name, mode='min', patience=50)
    # run_info = RunInfoCallback()

    # TRAINER
    trainer = L.Trainer(
        logger=wandb_logger,
        # callbacks=[model_ckpt, lr_monitor, early_stop, run_info],
        callbacks=[model_ckpt, lr_monitor],
        **asdict(config),
    )

    return trainer
