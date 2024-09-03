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

srun python  `pwd`/fno3d.py \
             --run 8 \
             --model_save_path="$BASE_REPO"/neural_operators \
             --train_data_path="$BASE_REPO"/RayleighBernardConvection/processed_data/RBC2D_NX256_NZ64_TI81_TF82_Pr1_Ra1_5e7_dt0_001_train.h5 \
             --val_data_path="$BASE_REPO"/RayleighBernardConvection/processed_data/RBC2D_NX256_NZ64_TI81_TF82_Pr1_Ra1_5e7_dt0_001_val.h5 \
             --train_samples=100 \
             --val_samples=50 \
             --input_timesteps=1 \
             --output_timesteps=1 \
             --start_index=0 \
             --stop_index=10 \
             --time_slice=1 \
             --dt=0.001 \
             --multi_step
             
            #  --load_checkpoint \
            #  --checkpoint_path="$BASE_REPO"/neural_operators/rbc_fno2d_time_N100_epoch3000_m12_w20_bs5_run5/checkpoint/model_checkpoint_999.pt


echo "END TIME: $(date)"
