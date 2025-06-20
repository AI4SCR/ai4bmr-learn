#!/bin/bash
#SBATCH --job-name=train-mae
#SBATCH --output=/users/amarti51/logs/train-mae-%A-%a.log
#SBATCH --error=/users/amarti51/logs/train-mae-%A-%a.err

#SBATCH --account=mrapsoma_prometex
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user amarti51@ad.unil.ch,adrianom@student.ethz.ch,adriano.martinelli@chuv.ch

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Load modules
module load python/3.12.1

PROJECT_DIR=/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ai4bmr-learn
cd "$PROJECT_DIR" || exit

/work/FAC/FBM/DBC/mrapsoma/prometex/envs/adrianom/rapids/bin/python3.12 /work/FAC/FBM/DBC/mrapsoma/prometex/projects/ai4bmr-learn/experiments/train_mae_cords_1d_conv.py
