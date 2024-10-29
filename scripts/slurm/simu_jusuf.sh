#!/bin/bash -x
#SBATCH --account=slmet

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --threads-per-core=1

#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --partition=develgpus
#SBATCH --disable-turbomode

# Setup environment
source ~/tlunet/setup.sh

# Go to script repo
cd -P ~/tlunet/neural_operators/scripts

srun -n 4 python -u 01_simu.py --dataDir=/p/scratch/cslmet/tlunet/simuData_dt1e-3 --dtData=0.001
