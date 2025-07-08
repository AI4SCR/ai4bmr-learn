#!/bin/bash
#SBATCH --job-name=maev1-sweep
#SBATCH --output=/users/amarti51/logs/ssl=maev1-sweep-%A-%a.log
#SBATCH --error=/users/amarti51/logs/ssl=maev1-sweep-%A-%a.err

#SBATCH --account=mrapsoma_prometex
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00

# Sweep 47 combinations → array indices 0..47
#SBATCH --array=0-47

# ------------------------------------------------------------------------------
# 0) Environment setup
# ------------------------------------------------------------------------------
source ~/miniconda3/bin/activate
conda activate beat
cd /work/FAC/FBM/DBC/mrapsoma/prometex/projects/ai4bmr-learn || exit

# ------------------------------------------------------------------------------
# 1) Define value‐lists for each hyperparameter
# ------------------------------------------------------------------------------
LR=(0.00015 0.0005 0.003)            # learning rate
WD=(0.04 0.02)                       # weight decay
BETAS=("[0.9,0.999]" "[0.95,0.999]") # betas as comma-separated strings
BS=(64 128)                          # batch size
AB=(1 8)                             # accumulate_grad_batches

# ------------------------------------------------------------------------------
# 2) Build an OVERRIDES array of full CLI‐strings
# ------------------------------------------------------------------------------
OVERRIDES=()
for lr in "${LR[@]}"; do
  for wd in "${WD[@]}"; do
    for betas in "${BETAS[@]}"; do
      for bs in "${BS[@]}"; do
        for ab in "${AB[@]}"; do
          OVERRIDES+=(
            "--model.lr=${lr} \
--model.weight_decay=${wd} \
--optimizer.betas=${betas} \
--data.batch_size=${bs} \
--trainer.accumulate_grad_batches=${ab}"
          )
        done
      done
    done
  done
done

# ------------------------------------------------------------------------------
# PRINT ALL OVERRIDES AND CHECK THAT ALL CONFIGS ARE VALID
# ------------------------------------------------------------------------------
echo "===== All override combos (total ${#OVERRIDES[@]}) ====="
for i in "${!OVERRIDES[@]}"; do
  echo "[$i] ${OVERRIDES[$i]}"
  python clis/maev1.py fit \
  --config configs/maev1-imagenet-vit.yaml \
  $ARGS \
  --print_config
done
echo "=============================================="

# ------------------------------------------------------------------------------
# 3) Select this task's override string
# ------------------------------------------------------------------------------
IDX=$SLURM_ARRAY_TASK_ID
ARGS=${OVERRIDES[$IDX]}

echo "[$(date)] Task \$IDX=$IDX → $ARGS"

# ------------------------------------------------------------------------------
# 4) Launch with the chosen overrides
# ------------------------------------------------------------------------------
python clis/dinov1.py fit \
  --config configs/dinov1-vit.yaml \
  $ARGS