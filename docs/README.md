# Neural Network based on Chebyshev-Fourier Neural Operator for Rayleigh-Benard Convection

- [**Base theory**](./theory/README.md)
- [**Usage documentation**](../scripts/README.md)

## Useful links

- [base FNO paper](https://arxiv.org/pdf/2010.08895)
- [Github FNO package](https://github.com/neuraloperator/neuraloperator)
- [DCT for torch](https://github.com/zh217/torch-dct)
- [SHNO paper](https://arxiv.org/pdf/2306.03838)
- [SHT for torch](https://github.com/NVIDIA/torch-harmonics)
- [reference code](https://github.com/neuraloperator/neuraloperator.git)

# Distributed Data Parallel (DDP)

Model training and inference can be deployed with distributed data parallel for faster processing of samples. Distributed Data Parallel (DDP) has been implemented using [PyTorch DDP](https://pytorch.org/docs/stable/notes/ddp.html#distributed-data-parallel) where the model is replicated and trained on different samples followed by synchronisation of weights.
To enable DDP set the following parameters in the `config.yaml` used:
```
parallel_strategy:
  ddp: True              # enable Distributed Data Parallel
  gpus_per_node: 4
```
DDP job on GPUs is launched using `torchrun`  with SLURM as:
```
##### Network parameters #####
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
GPUS_PER_NODE=4

srun python -u -m torch.distributed.run  \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_JOB_NUM_NODES \
    --node_rank $SLURM_PROCID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --rdzv_conf=is_host=$(if ((SLURM_NODEID)); then echo False; else echo True; fi) \
    --max_restarts 0 \
    --tee 3 \
    $python_file.py \
    --config config.yaml
```
For training `$python_file.py` is [03_train.py](../scripts/03_train.py).
