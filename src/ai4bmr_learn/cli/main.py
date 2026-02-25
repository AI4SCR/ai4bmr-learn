import lightning
import torch
from lightning.pytorch.cli import LightningCLI

from dotenv import load_dotenv
from ai4bmr_learn.callbacks.save_config import LoggerSaveConfigCallback

# ENVIRONMENT MANAGEMENT
load_dotenv()
torch.set_float32_matmul_precision('medium')

def cli_main():
    cli = LightningCLI(lightning.LightningModule, lightning.LightningDataModule,
                       subclass_mode_data=True, subclass_mode_model=True, parser_kwargs={"parser_mode": "omegaconf"},
                       save_config_callback=LoggerSaveConfigCallback, save_config_kwargs={"save_to_log_dir": False})


if __name__ == "__main__":
    cli_main()
