#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --job-name=FNO2D
#SBATCH --account=exalab
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                   
#SBATCH --ntasks-per-node=1             
#SBATCH --cpus-per-task=48             
#SBATCH --time=02:00:00               
#SBATCH --threads-per-core=1            
#SBATCH --output=%x-%j.out              

# explicitly setting srun environment variable to inherit from SBATCH
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

export BASE_REPO="/p/project1/cexalab/john2/NeuralOperators"

# Enable logging
# set -euo pipefail
# set -x

cd "$BASE_REPO"/neural_operators
source setup.sh
echo "START TIME: $(date)"

srun python `pwd`/fnop/scripts/train_fno2d_recurrent.py --config_file `pwd`/fnop/configs/fno2d.yaml
# srun python `pwd`/fnop/scripts/train_fno3d.py --config_file `pwd`/fnop/configs/fno3d.yaml

echo "END TIME: $(date)"


