#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --job-name=FNO2D_recurtime
#SBATCH --account=exalab
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                   
#SBATCH --ntasks-per-node=1             
#SBATCH --cpus-per-task=48             
#SBATCH --time=03:30:00               
#SBATCH --threads-per-core=1            
#SBATCH --output=%x-%j.out              

# explicitly setting srun environment variable to inherit from SBATCH
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

export BASE_REPO="/p/project1/cexalab/john2/NeuralOperators"

# Enable logging
# set -euo pipefail
# set -x

source setup.sh
cd "$BASE_REPO"/neural_operators
echo "START TIME: $(date)"

srun python  `pwd`/fno2d_recurrent.py \
             --run 5 \
             --model_save_path="$BASE_REPO"/neural_operators \
             --train_data_path="$BASE_REPO"/RayleighBernardConvection/processed_data/RBC2D_NX256_NZ64_TI81_TF82_Pr1_Ra1_5e7_dt0_001_train.h5 \
             --val_data_path="$BASE_REPO"/RayleighBernardConvection/processed_data/RBC2D_NX256_NZ64_TI81_TF82_Pr1_Ra1_5e7_dt0_001_val.h5 \
             --test_data_path="$BASE_REPO"/RayleighBernardConvection/processed_data/RBC2D_NX256_NZ64_TI80_TF83_Pr1_Ra1_5e7_dt0_001_test.h5 \
             --load_checkpoint \
             --checkpoint_path="$BASE_REPO"/neural_operators/rbc_fno2d_time_N100_epoch3000_m12_w20_bs5_run5/checkpoint/model_checkpoint_999.pt
             
echo "END TIME: $(date)"

# srun python  `pwd`/fno3d.py \
#              --run 5 \
#              --model_save_path="$BASE_REPO"/neural_operators \
#              --data_path="$BASE_REPO"/RayleighBernardConvection/processed_data/RBC2D_NX256_NZ64_TI0_TF150_Pr1_Ra1e7_dt0_1.h5 \
#              --exit-duration-in-mins 60 \
#              --load_checkpoint \
#              --checkpoint_path="$BASE_REPO"/neural_operators/rbc_fno3d_N100_epoch500_m12_w32_bs5_run2/checkpoint/model_checkpoint_120.pt