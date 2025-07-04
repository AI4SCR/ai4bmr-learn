#!/bin/bash
#SBATCH --job-name=proj=beat-task=compute-embeddings
#SBATCH --output=/users/amarti51/logs/proj=beat-task=compute-embeddings-%j.log
#SBATCH --error=/users/amarti51/logs/proj=beat-task=compute-embeddings-%j.err

#SBATCH --account=mrapsoma_prometex

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Load modules
source ~/miniconda3/bin/activate
conda activate beat

python /work/FAC/FBM/DBC/mrapsoma/prometex/projects/ai4bmr-learn/slurm/scripts/beat-compute-embeddings.py
