#!/bin/bash
#SBATCH --job-name=cooperative_sheaves
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-49

source /home/parentd1/.bashrc
conda activate cooperative_sheaves

export WANDB_API_KEY=0966ca370400b75fea100ca719360624e659271f
export WANDB__SERVICE_WAIT=300

SWEEP_IDS=(
  "juanbelieni/coopshv_sweeps/n0rmwhna" # Amazon ratings
  "juanbelieni/coopshv_sweeps/nylv1ymp" # Roman Empire
  "juanbelieni/coopshv_sweeps/dxqkehf6" # Questions
  "juanbelieni/coopshv_sweeps/vx4irq3d" # Tolokers
  "juanbelieni/coopshv_sweeps/9m5j6g0x" # Minesweeper
)

NUM_SWEEPS=${#SWEEP_IDS[@]}

IDX=$(( SLURM_ARRAY_TASK_ID % NUM_SWEEPS ))
export SWEEP_ID=${SWEEP_IDS[$IDX]}

wandb agent "$SWEEP_ID"
