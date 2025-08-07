# AI4BMR-LEARN Project Analysis

## Project Overview

This project, `ai4bmr-learn`, is a Python-based machine learning framework for biomedical research. It appears to be focused on applying various deep learning techniques, particularly for image analysis and classification tasks, as suggested by the presence of libraries like `lightly` (for self-supervised learning), `timm` (for image models), and `openslide-python` (for whole-slide image reading). The framework is built upon PyTorch and Lightning, utilizing a configuration-driven approach for experiments.

## Building and Running

### Dependencies

The project uses Poetry for dependency management. To install the required packages, run:

```bash
poetry install
```

### Running Experiments

The main entry point for running experiments is `clis/main.py`, which utilizes the `lightning.pytorch.cli.LightningCLI`. This allows for a flexible and configurable way to run training and evaluation pipelines.

To run an experiment, you would typically use a command like this:

```bash
python clis/main.py fit --config configs/your_config.yaml
```

For example, to run a debugging configuration:

```bash
python clis/main.py fit --config configs/debug.yaml
```

The `configs` directory contains various YAML files for different experiments, such as `clf-cords2024-resnet.yaml` or `dinov1-imagenet-vit.yaml`. These files define the model, data, and trainer configurations.

## Development Conventions

*   **Code Formatting:** The project uses `black` for code formatting, as indicated in the `pyproject.toml` file.
*   **Configuration:** Experiments are configured using YAML files, which are then parsed by the `LightningCLI`. This promotes reproducibility and easy modification of experiment parameters.
*   **Environment Variables:** The project uses a `.env` file for environment variable management, as seen in `clis/main.py`. This is likely used for storing secrets like API keys for services such as Weights & Biases (`wandb`).
*   **Testing:** A `tests` directory exists, suggesting that the project has a suite of tests.

## Project Structure

*   `src/ai4bmr_learn`: Contains the core source code for the project, including data modules, models, callbacks, and utilities.
*   `configs`: Contains YAML configuration files for various experiments.
*   `clis`: Contains the main entry points for the command-line interface.
*   `experiments`: Contains scripts for running specific experiments.
*   `scripts`: Contains utility scripts for tasks like preparing datasets or extracting model weights.
*   `tests`: Contains tests for the project.
*   `slurm`: Contains scripts for running jobs on a Slurm cluster.
*   `pyproject.toml`: Defines the project dependencies and metadata.
*   `requirements.txt`: Lists the project dependencies.
*   `README.md`: Provides a brief overview of the project and its limitations.
