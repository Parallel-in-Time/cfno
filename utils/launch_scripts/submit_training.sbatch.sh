#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --job-name=training
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
export BASE_REPO="cfno"

source "$BASE_REPO"/utils/setup.sh
echo "START TIME: $(date)"
cd "$BASE_REPO"

srun python `pwd`/cfno/train_scripts/train_cfno2d.py --config_file `pwd`/cfno/configs/cfno2d.yaml

echo "END TIME: $(date)"

