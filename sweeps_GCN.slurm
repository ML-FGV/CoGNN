#!/bin/bash
#SBATCH --job-name=CoGNN
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-99

source /home/parentd1/.bashrc
conda activate cooperative_sheaves

export WANDB_API_KEY=0966ca370400b75fea100ca719360624e659271f
export WANDB__SERVICE_WAIT=300

SWEEP_IDS=(
  # "juanbelieni/coopshv_sweeps/n0rmwhna" # Amazon ratings
  # "juanbelieni/coopshv_sweeps/nylv1ymp" # Roman Empire
  # "juanbelieni/coopshv_sweeps/dxqkehf6" # Questions
  # "juanbelieni/coopshv_sweeps/vx4irq3d" # Tolokers
  # "juanbelieni/coopshv_sweeps/9m5j6g0x" # Minesweeper
  # "andrerg00/CoGNN/cy9eti9v" # Chameleon
  # "andrerg00/CoGNN/avam8ee1" # Squirrel
  # "andrerg00/CoGNN/yyoa0ntr" # Minesweeper with GPS
  # "andrerg00/CoGNN_GCN/s7a9hkgt" # Amazon ratings GCN
  # "andrerg00/CoGNN_GCN/gq0cu6d6" # Minesweeper GCN
  # "andrerg00/CoGNN_GCN/4ou37zta" # Roman Empire GCN
    "andrerg00/CoGNN_NotStarGCN/wj44eu15" # Minesweeper
    "andrerg00/CoGNN_NotStarGCN/m0jden2n" # Roman empire
)

NUM_SWEEPS=${#SWEEP_IDS[@]}

IDX=$(( SLURM_ARRAY_TASK_ID % NUM_SWEEPS ))
export SWEEP_ID=${SWEEP_IDS[$IDX]}

wandb agent "$SWEEP_ID"
