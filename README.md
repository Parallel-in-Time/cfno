# Neural Operators for learning of PDEs

This repository contains implementation of [Fourier Neural Operators](https://arxiv.org/abs/2010.08895).

## Requirements

The code is based on python3 (version 3.11) and the packages required can be installed with

```bash
python3 -m pip install -r requirements.txt
```

The code is developed into a python package allowing to do the model training, validation and inference :
`fnop`. You can install it on your system, without root rights, by doing this in this repository :

```bash
pip install -e --user .
```

> The `-e` option installs in _editable_ mode, which means any modification in the `fnop` code won't need a re-installation to take the change into account.

To de-install the package, simply run :

```bash
pip uninstall fnop
```

## Data

For 2D Rayleigh Benard Convection (RBC) problem the data is generated using [Dedalus](https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_2d_rayleigh_benard.html).

See also the [dedalus](./dedalus/) folder.


## Model Training & Inference

[Fourier Neural Operator 2D Spatial + Recurrent in time](./fnop/models/fno2d_recurrent.py) and [Fourier Neural Operator 2D Spatial + 1D time](./fnop/models/fno3d.py) solver for RBC 2D.

Example submission script to train on JUWELS Booster is shown in [submit_juwels.sbatch](./launch_scripts/submit_juwels.sbatch.sh).

Model inference can be done using [inference.py](./fnop/inference/inference.py), see example submission script [submit_inference.sbatch.sh](./launch_scripts/submit_inference.sbatch.sh).

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