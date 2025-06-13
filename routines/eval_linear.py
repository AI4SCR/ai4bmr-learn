from lightning.pytorch.cli import LightningCLI

# simple demo classes for your convenience
from ai4bmr_learn.datamodules.Tabular import TabularDataModule
from ai4bmr_learn.routines.linear_probing import LinearProbing

def cli_main():
    cli = LightningCLI(model_class=LinearProbing, datamodule_class=TabularDataModule)


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block