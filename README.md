# Neural Operators for learning of PDEs

This repository contains implementation of [Fourier Neural Operators](https://arxiv.org/abs/2010.08895)


## Requirements
The code is based on python3 (version 3.11) and the packages required can be installed with
```bash
	python3 -m pip install -r requirements.txt
```

## Data

For 2D Rayleigh Benard Convection (RBC) problem the data is generated using [Dedalus](https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_2d_rayleigh_benard.html)

See also the [dedalus](../dedalus/) folder


## Model Training & Inference

[Fourier Neural Operator 2D Spatial + Recurrent in time](./fno2d_recurrent.py) and [Fourier Neural Operator 2D Spatial + 1D time](./fno3d.py) solver for RBC 2D

Example submission script to train on JUWELS Booster is shown in [submit_juwels.sbatch](./submit_juwels.sbatch.sh)

Model inference can be done using [inference.py](./inference.py)

## Note

- Training can be done using CPU or a single GPU.

- GPU testing done on JUWELS Booster: 1X NVIDIA A100 (40GB) GPU with:
	- GCC-12.3.0 
	- OpenMPI-4.1.5 
	- CUDA-12 

- Python environment activated using [setup.sh](./setup.sh)
- [examples](./examples/): Contains files for solving 
	- Wave equation
	- Darcy Flow equation
	- Rayleigh Benard Convection 2D on 64 x 64 grid for Ra=10000 
	
## Reference:

- [neural_operator](https://github.com/neuraloperator/neuraloperator.git)