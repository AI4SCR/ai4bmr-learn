import lightning
import torch
from lightning.pytorch.cli import LightningCLI

from ai4bmr_learn.supervised.classification import Classifier
from dotenv import load_dotenv
from ai4bmr_learn.utils.utils import setup_wandb_auth
from ai4bmr_learn.callbacks.save_config import LoggerSaveConfigCallback

# ENVIRONMENT MANAGEMENT
load_dotenv()
setup_wandb_auth()
torch.set_float32_matmul_precision('medium')

def cli_main():
    cli = LightningCLI(lightning.LightningModule, lightning.LightningDataModule, subclass_mode_data=True,
                       save_config_callback=LoggerSaveConfigCallback, save_config_kwargs={"save_to_log_dir": False})


if __name__ == "__main__":
    cli_main()