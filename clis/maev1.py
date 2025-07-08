import lightning
import torch
from lightning.pytorch.cli import LightningCLI

from ai4bmr_learn.ssl.maev1 import MAEv1
from dotenv import load_dotenv
from ai4bmr_learn.utils.utils import setup_wandb_auth

# ENVIRONMENT MANAGEMENT
load_dotenv()
setup_wandb_auth()
torch.set_float32_matmul_precision('medium')

def cli_main():
    cli = LightningCLI(MAEv1, lightning.LightningDataModule,
                       subclass_mode_data=True, save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    cli_main()