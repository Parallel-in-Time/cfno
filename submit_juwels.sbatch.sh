#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --job-name=FNO_2D_recur
#SBATCH --account=exalab
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                   
#SBATCH --ntasks-per-node=1             
#SBATCH --cpus-per-task=48             
#SBATCH --time=01:00:00               
#SBATCH --threads-per-core=1            
#SBATCH --output=%x-%j.out              

# explicitly setting srun environment variable to inherit from SBATCH
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

export BASE_REPO="/p/project1/cexalab/john2/NeuralOperators"

# Enable logging
set -euo pipefail
set -x

source setup.sh
cd "$BASE_REPO"/neural_operators
echo "START TIME: $(date)"

srun python  `pwd`/fno2d_recurrent.py \
             --model_save_path="$BASE_REPO"/neural_operators \
             --data_path="$BASE_REPO"/RayleighBernardConvection/processed_data/RBC2D_NX256_NZ64_TI0_TF150_Pr1_Ra1e7_dt0_1.h5
             
echo "END TIME: $(date)"