from dataclasses import asdict

import wandb
from loguru import logger

from ai4bmr_learn.data_models import WandbInitConfig


def setup_wandb(wandb_init: WandbInitConfig, config=None):
    if wandb.run is not None:
        logger.info(f"Active wandb run found ({wandb.run.name}). Skipping wandb init.")
        return True
    else:
        wandb.init(**asdict(wandb_init), config=config)
        return False
