#!/bin/bash
#SBATCH --job-name=dinov1-lightly
#SBATCH --output=/users/amarti51/logs/ssl=dinov1-lightly-%j.log
#SBATCH --error=/users/amarti51/logs/ssl=dinov1-lightly-%j.err

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


for i in {1..3}; do
    #echo "$i th resnet18 iteration"
    #python /work/FAC/FBM/DBC/mrapsoma/prometex/projects/ai4bmr-learn/examples/dinov1-lightly-resnet18.py
    #echo "$i th vit iteration"
    #python /work/FAC/FBM/DBC/mrapsoma/prometex/projects/ai4bmr-learn/examples/dinov1-lightly-vit.py
    #echo "$i th vit iteration"
    #python /work/FAC/FBM/DBC/mrapsoma/prometex/projects/ai4bmr-learn/examples/dinov1-lightly-vit.py
done