#!/bin/bash
#SBATCH --job-name=prepare-beat
#SBATCH --output=/users/amarti51/logs/prepare-beat-%j.log
#SBATCH --error=/users/amarti51/logs/prepare-beat-%j.err

#SBATCH --account=mrapsoma_prometex
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=amarti51@ad.unil.ch,adrianom@student.ethz.ch,adriano.martinelli@chuv.ch

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=8:00:00

# Load environment
source /users/amarti51/miniconda3/bin/activate
conda activate beat

# Change to project directory
PROJECT_DIR="/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ai4bmr-learn"
cd "$PROJECT_DIR" || exit 1

# Run the preparation script
python slurm/scripts/prepare-beat.py