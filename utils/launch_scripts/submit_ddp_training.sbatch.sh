#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --job-name=ddp_training
#SBATCH --account=exalab
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --gres=gpu:4                   
#SBATCH --ntasks-per-node=1             
#SBATCH --cpus-per-task=48            
#SBATCH --time=0:30:00               
#SBATCH --threads-per-core=1            
#SBATCH --output=%x-%j.out              
#SBATCH --error=%x-%j.err


# explicitly setting srun environment variable to inherit from SBATCH
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# Enable logging
set -euo pipefail
set -x

##### Network parameters #####
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# Allow communication over InfiniBand cells.
MASTER_ADDR="${MASTER_ADDR}i"
MASTER_PORT=6000
GPUS_PER_NODE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
export BASE_REPO="neural_operators"
export NCCL_SOCKET_IFNAME=ib0
# export TORCH_USE_CUDA_DSA=1

source "$BASE_REPO"/utils/setup.sh
echo "START TIME: $(date)"
cd "$BASE_REPO"

srun --jobid $SLURM_JOBID python -u -m torch.distributed.run  \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_JOB_NUM_NODES \
    --node_rank $SLURM_PROCID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --rdzv_conf=is_host=$(if ((SLURM_NODEID)); then echo False; else echo True; fi) \
    --max_restarts 0 \
    --tee 3 \
    `pwd`/scripts/03_train.py \
    --config `pwd`/scripts/config.yaml

echo "END TIME: $(date)"

