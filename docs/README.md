# Some docs on the FNO and CFNO stuff ...

## Generated

- [fno.svg](./fno.svg) : graphical representation of a 1D FNO with one Fourier layer
- [fno2d.svg](./fno2d.svg) : graphical representation of a 2D FNO with one Fourier layer
- [cfno2d.svg](./cfno2d.svg): graphical representation of a 2D Chebyshev-Fourier Neural Operator (CFNO) with one Fourier layer 

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
For training `$python_file.py` is [03_train.py](../scripts/03_train.py) and for inference it is [04_eval.py](../scripts/04_eval.py).

Example submission scripts for JSC systems can be found in [launch_scripts](../utils/launch_scripts/) folder.

For inference, the initial input to DDP model setting is given through DataLoader with DistributedSampler such that `nSamples` are divided equally among the participating GPUs and the batch_size is set as `nSamples/nGPUs` so that there is only single pass through the DataLoader. This is done since the current models are very small and would be efficient to process as many samples at once as possible.

For performing `nEval` > 1 , the input to the DDP model is the output of the DDP model at previous iteration. Once the evaluation is completed the output of the DDP models are concatenated using `allgather` to form the complete output.


## Code for the original approach (old documentation)

### Model Training

For 2D Rayleigh Benard Convection (RBC) problem the data is generated using [Dedalus](https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_2d_rayleigh_benard.html). See also the [dedalus](./dedalus/) folder.

[Fourier Neural Operator 2D Spatial + Recurrent in time](../cfno/models/fno2d.py) and [Fourier Neural Operator 2D Spatial + 1D time](../cfno/models/fno3d.py) solver for RBC 2D.

Example submission scripts to train on JUWELS Booster is shown in [submit_training.sbatch.sh](../utils/launch_scripts/submit_training.sbatch.sh).

### Model Inference

Once a model is trained, it is saved into a _checkpoint_ file, that are stored in the [`model_archive` companion repo](https://codebase.helmholtz.cloud/neuralpint/model_archive) as `*.pt` files 
(weights of the model).
Which each of those models is associated a YAML configuration file, that stores all the model setting
parameters.

One given trained model should be used like this :

```python
from fnop.inference import FNOInference

# Load the trained model
model = FNOInference(
    config="path/to/configFile.yaml",
    checkpoint="path/to/checkpointFile.pt")

u0 = ... # some solution of RBC at a given time
# --> numpy.ndarray format, with shape (nVar, nX, nZ), with nVar = 4

# In particular, uInit could be unpacked like this :
vx0, vy0, b0, p0 = u0

# Evaluate the model with a given initial solution using a fixed time-step
u1 = model.predict(u0)  # -> only one time-step !

# --> u1 can also be unpacked like u0, e.g :
vx1, vy1, b1, p1 = u1

# Time-step of the model can be retrieved like this :
dt = model.dt 	# -> can be either an attribute or a property
```

