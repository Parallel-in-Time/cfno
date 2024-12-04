#!/bin/bash -l
#SBATCH --job-name=ddpmodel_eval
#SBATCH --account=exalab  
#SBATCH --partition=dc-gpu-devel          # change partition
#SBATCH --cpus-per-task=48
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --threads-per-core=1    
#SBATCH --time=00:25:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

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
export NCCL_SOCKET_IFNAME=ib0

echo "START TIME: $(date)"
cd neural_operators
### To run evaluation without DDP
# srun  python ./scripts/04_eval.py --config=./scripts/config.yaml

### To run evaluation with DPP
srun --jobid $SLURM_JOBID python -u -m torch.distributed.run  \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_JOB_NUM_NODES \
    --node_rank $SLURM_PROCID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --rdzv_conf=is_host=$(if ((SLURM_NODEID)); then echo False; else echo True; fi) \
    --max_restarts 0 \
    --tee 3 \
    ./scripts/04_eval.py \
    --config=./scripts/ddp_config.yaml

echo "END TIME: $(date)"
