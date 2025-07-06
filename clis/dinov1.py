import torch
from lightning.pytorch.cli import LightningCLI

from ai4bmr_learn.datamodules.imagenet import ImageNet
from ai4bmr_learn.ssl.dinov1 import DINOv1
from dotenv import load_dotenv
from ai4bmr_learn.utils.utils import setup_wandb_auth

# ENVIRONMENT MANAGEMENT
load_dotenv()
setup_wandb_auth()
torch.set_float32_matmul_precision('medium')

def cli_main():
    cli = LightningCLI(DINOv1, ImageNet, save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    cli_main()