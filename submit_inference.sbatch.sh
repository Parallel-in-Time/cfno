#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --job-name=FNO_inference
#SBATCH --account=exalab
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                   
#SBATCH --ntasks-per-node=1             
#SBATCH --cpus-per-task=48             
#SBATCH --time=00:20:00               
#SBATCH --threads-per-core=1            
#SBATCH --output=%x-%j.out              

# explicitly setting srun environment variable to inherit from SBATCH
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

export BASE_REPO="/p/project1/cexalab/john2/NeuralOperators"

source setup.sh
cd "$BASE_REPO"/neural_operators
echo "START TIME: $(date)"
             
#--cpu-bind=v
srun python inference.py \
             --folder=/p/project1/cexalab/john2/NeuralOperators/neural_operators/rbc_fno2d_time_N100_epoch3000_m12_w20_bs5_tin1_run6 \
             --model=/p/project1/cexalab/john2/NeuralOperators/neural_operators/rbc_fno2d_time_N100_epoch3000_m12_w20_bs5_tin1_run6/checkpoint/model_checkpoint_2999.pt \
             --test_data_path="$BASE_REPO"/RayleighBernardConvection/processed_data/RBC2D_NX256_NZ64_TI80_TF83_Pr1_Ra1_5e7_dt0_001_test.h5 \
             --dim=FNO2D \
             --modes=12 \
             --width=20 \
             --batch_size=1 \
             --time_file=/p/scratch/cexalab/john2/RBC2D_data_dt1e_3/RBC2D_NX256_NZ64_TI0_TF150_Pr1_Ra1_5e7_dt0_001_test_5/RBC2D_NX256_NZ64_TI0_TF150_Pr1_Ra1_5e7_dt0_001_test_5_s1.h5
          

# python inference.py \
#              --model="$BASE_REPO"/neural_operators/results/rbc_fno_3d_N100_epoch500_m8_w20_bs5/checkpoint/model_checkpoint_400.pt \
#              --data_path="$BASE_REPO"/RayleighBernardConvection/processed_data/RBC2D_NX256_NZ64_TI0_TF150_Pr1_Ra1e7_dt0_1.h5 \
#              --modes=8 \
#              --width=20 \
#              --batch_size=50 \
#              --folder="$BASE_REPO"/neural_operators/results  \
#              --time_file="$BASE_REPO"/RayleighBernardConvection/data/RBC2D_NX256_NZ64_TI0_TF150_Pr1_Ra1e7_dt0_1_test_1/RBC2D_NX256_NZ64_TI0_TF150_Pr1_Ra1e7_dt0_1_test_1_s1.h5 \
#              --plotFile  

echo "END TIME: $(date)"