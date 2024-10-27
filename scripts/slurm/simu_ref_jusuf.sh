#!/bin/bash -x
#SBATCH --account=slmet

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --threads-per-core=1

#SBATCH --time=19:00:00
#SBATCH --exclusive
#SBATCH --partition=batch
#SBATCH --disable-turbomode

# Setup environment
source ~/tlunet/setup.sh

# Go to script repo
cd -P ~/tlunet/neural_operators/scripts

srun -n 4 python -u 01_simu.py --dataDir=/p/scratch/cslmet/tlunet/simuData_dt1e-3_ref --dtData=0.001 --dtSimu=0.00005
