#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --job-name=FNO_2D
#SBATCH --account=exalab
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                   
#SBATCH --ntasks-per-node=1             
#SBATCH --cpus-per-task=48             
#SBATCH --time=04:00:00               
#SBATCH --threads-per-core=1            
#SBATCH --output=%x-%j.out              

# explicitly setting srun environment variable to inherit from SBATCH
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

export BASE_REPO="cfno"

# Enable logging
set -euo pipefail
set -x

source "$BASE_REPO"/utils/setup.sh
cd "$BASE_REPO"/docs/examples
echo "START TIME: $(date)"

srun python  `pwd`/train_FNO.py \
             --problem="wave" \
             --model_save_path="$SYSTEMNAME"_run1
             
echo "END TIME: $(date)"