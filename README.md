# Neural Operators for learning of PDEs

This repository contains implementation of [Fourier Neural Operators](https://arxiv.org/abs/2010.08895)


## Requirements
The code is based on python 3 (version 3.11) and the packages required can be installed with

	python3 -m pip install -r requirements.txt


## Data
This code base focuses on the Rayleigh-Bénard convection (RBC) problem, as a starting point wave equation and darcy flow is solved using the data downloaded from https://zenodo.org/records/10406879 (~2.4GB). It covers instances of the Poisson, Wave, Navier-Stokes, Allen-Cahn, Transport and Compressible Euler equations and Darcy flow. 

Run the script `download_data.py` which downloads all required data into the appropriate folder (it requires 'wget' to be installed on your system).

For RBC problem the data is generated using [Dedalus](https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_2d_rayleigh_benard.html)

## Models Training

To run on JUWELS booster use `submit.sbatch.sh` file that activates the python virtual environment and loads the required modules using `setup.sh` and executes the  `train_FNO.py` file.

`FNO.ipynb` jupyter notebook is also available.

To select problem, the variable "which_example" in `train_FNO.py` should have one of the following values:
```
    wave                : Wave equation
    darcy               : Darcy Flow
    RBC2D               : Rayleigh-Bénard convection 2D
```

## Note

Training can be done using CPU or a single GPU.

GPU testing done on JUWELS Booster: 1X NVIDIA A100 (40GB) GPU with:
- GCC-12.3.0 
- OpenMPI-4.1.5 
- CUDA-12 

The following files correspond to:

```
    problems.py : Dataloader for FNO model
    fourier_operator: FNO modules
    utils.py : Sumplemment functions
    DataVisual.ipynb: Data preprocessing
    inference.py: Testing trained model
```

## Reference:
- [Convolutional Operators Code](https://github.com/bogdanraonic3/ConvolutionalNeuralOperator)
- [Neural Operators code](https://github.com/neuraloperator/neuraloperator)
