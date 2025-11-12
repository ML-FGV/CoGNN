#!/bin/bash
#SBATCH --job-name=CoGNN
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-49

source /home/parentd1/.bashrc
conda activate cooperative_sheaves

export WANDB_API_KEY=ab6e83a9d99012b14cff411df4e314597d9e6898
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
  andrerg00/CoGNN_small_heterophilous/x08vg56p
  andrerg00/CoGNN_small_heterophilous/hpyyvyyg
  andrerg00/CoGNN_small_heterophilous/tjslc22v
  andrerg00/CoGNN_small_heterophilous/mf1sevuz
)

NUM_SWEEPS=${#SWEEP_IDS[@]}

IDX=$(( SLURM_ARRAY_TASK_ID % NUM_SWEEPS ))
export SWEEP_ID=${SWEEP_IDS[$IDX]}

wandb agent "$SWEEP_ID"
