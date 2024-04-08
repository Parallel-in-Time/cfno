#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --job-name=fno2d_RBC
#SBATCH --account=exalab
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                    # numbe of GPUs
#SBATCH --ntasks-per-node=1             
#SBATCH --cpus-per-task=48              # Slurm 22.05: srun doesnot inherit this variable from sbatch
#SBATCH --time=02:00:00                 # maximum execution time (HH:MM:SS)
#SBATCH --threads-per-core=1            # using only real cores, no SMT
#SBATCH --error=%x-%j.log               # log file name

# explicitly setting srun environment variable to inherit from SBATCH
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

export BASE_REPO="/p/project/cexalab/john2/NeuralOperators/neural_operators"

# Enable logging
set -euo pipefail
set -x

source setup.sh
cd "$BASE_REPO"
echo "START TIME: $(date)"

srun python  `pwd`/train_FNO.py RBC2D

echo "END TIME: $(date)"