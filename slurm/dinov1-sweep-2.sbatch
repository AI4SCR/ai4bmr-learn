#!/bin/bash
#SBATCH --job-name=dinov1-sweep-v2
#SBATCH --output=/users/amarti51/logs/ssl=dinov1-sweep-v2-%A_%a.log
#SBATCH --error=/users/amarti51/logs/ssl=dinov1-sweep-v2-%A_%a.err

#SBATCH --account=mrapsoma_prometex
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00

# Sweep 32 combinations → array indices 0..31
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
LR=(0.003 0.0005 0.0001) # lr
WD=(0.04 0.02)           # weight_decay
TT=(0.04 0.06)           # teacher_temp
MOM=(0.996 0.998)        # momentum
NOR=(True)               # norm_last_layer
FRZ=(3)                  # freeze_last_layer
BS=(64, 128)             # data.batch_size

# ------------------------------------------------------------------------------
# 2) Build an OVERRIDES array of full CLI‐strings
# ------------------------------------------------------------------------------
OVERRIDES=()
for lr in "${LR[@]}"; do
  for wd in "${WD[@]}"; do
    for tt in "${TT[@]}"; do
      for mom in "${MOM[@]}"; do
        for nor in "${NOR[@]}"; do
          for frz in "${FRZ[@]}"; do
            for bs in "${BS[@]}"; do
              OVERRIDES+=(
              "--model.lr=${lr} \
--model.weight_decay=${wd} \
--model.teacher_temp=${tt} \
--model.momentum=${mom} \
--model.norm_last_layer=${nor} \
--model.freeze_last_layer=${frz} \
--data.batch_size=${bs}"
          )
            done
          done
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
  #python clis/dinov1.py fit \
  #--config configs/dinov1-vit.yaml \
  #$ARGS \
  #--print_config
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