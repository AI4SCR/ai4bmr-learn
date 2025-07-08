#!/bin/bash
#SBATCH --job-name=dinov1-sweep
#SBATCH --output=/users/amarti51/logs/ssl=dinov1-%A-%a.log
#SBATCH --error=/users/amarti51/logs/ssl=dinov1-%A-%a.err

#SBATCH --account=mrapsoma_prometex
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:59:59

# Update this based on number of combinations
#SBATCH --array=0-287

# ------------------------------------------------------------------------------
# 0) Environment setup
# ------------------------------------------------------------------------------
source ~/miniconda3/bin/activate
conda activate beat
cd /work/FAC/FBM/DBC/mrapsoma/prometex/projects/ai4bmr-learn || exit

# ------------------------------------------------------------------------------
# 1) Define value‐lists for each hyperparameter
# ------------------------------------------------------------------------------
LR=(0.005 0.003 0.0005)    # learning rate
WD=(0.04 0.02)             # weight decay
TT=(0.04 0.06 0.08)        # teacher_temp
MOM=(0.996 0.998)          # momentum
NOR=(True False)           # norm_last_layer
FRZ=(1 3)                  # freeze_last_layer
BS=(64 128)                # batch size
AB=(1)                     # accumulate_grad_batches

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
              for ab in "${AB[@]}"; do
                OVERRIDES+=(
                  "--model.lr=${lr} \
--model.weight_decay=${wd} \
--model.teacher_temp=${tt} \
--model.momentum=${mom} \
--model.norm_last_layer=${nor} \
--model.freeze_last_layer=${frz} \
--data.init_args.batch_size=${bs} \
--trainer.accumulate_grad_batches=${ab}"
                )
              done
            done
          done
        done
      done
    done
  done
done

# ------------------------------------------------------------------------------
# PRINT ALL OVERRIDES AND DETERMINE ARRAY SIZE
# ------------------------------------------------------------------------------
echo "===== All override combos (total ${#OVERRIDES[@]}) ====="
for i in "${!OVERRIDES[@]}"; do
  echo "[$i] ${OVERRIDES[$i]}"
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
  --config configs/dinov1-imagenet-resnet.yaml \
  $ARGS