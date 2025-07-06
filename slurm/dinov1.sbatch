#!/bin/bash
#SBATCH --job-name=dinov1
#SBATCH --output=/users/amarti51/logs/ssl=dinov1-%j.log
#SBATCH --error=/users/amarti51/logs/ssl=dinov1-%j.err

#SBATCH --account=mrapsoma_prometex

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Load modules
source ~/miniconda3/bin/activate
conda activate beat

PROJECT_DIR=/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ai4bmr-learn
cd "$PROJECT_DIR" || exit

# higher teacher temp 0.04 -> 0.07
# higher teacher momentum 0.996 -> 0.999
# norm_last_layer True -> False
# freeze_last_layer 1 -> 3
# batch_size 64 -> 128


python /work/FAC/FBM/DBC/mrapsoma/prometex/projects/ai4bmr-learn/clis/dinov1.py fit --config configs/dinov1-vit.yaml