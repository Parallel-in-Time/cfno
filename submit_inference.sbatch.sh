#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --job-name=FNO_inference
#SBATCH --account=exalab
#SBATCH --partition=dc-gpu-devel
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
             
# srun --cpu-bind=none,v  
python inference.py \
        --run=1 \
        --model_checkpoint=/p/scratch/cexalab/john2/trial_runs_strategy2_tin10/rbc_fno2d_time_N100_epoch3000_m12_w20_bs10_dt1e_3_run4/checkpoint/model_checkpoint_2999.pt \
        --test_data_path=/p/project1/cexalab/john2/NeuralOperators/RayleighBernardConvection/processed_data/RBC2D_NX256_NZ64_TI80_TF83_Pr1_Ra1_5e7_dt0_001_test.h5 \
        --dim=FNO2D \
        --modes=12 \
        --width=20 \
        --batch_size=1 \
        --rayleigh=1.5e7 \
        --prandtl=1.0\
        --gridx=256 \
        --gridy=64 \
        --nTest=1 \
        --T_in=10 \
        --T=1 \
        --start_index=0 \
        --stop_index=3000 \
        --timestep=1000 \
        --dedalus_time_index=80000 \
        --dt=0.001 \
        --folder=/p/project1/cexalab/john2/NeuralOperators/neural_operators  \
        --calc_loss \
        --store_result

        # --plotFile
        # --train_data_path=<train_data_path> (only when using FNO3D) \
        # --train_samples
  

echo "END TIME: $(date)"