# Neural Operators for learning of PDEs

This repository contains first implementation of [Fourier Neural Operators](https://arxiv.org/abs/2010.08895)


## Requirements
The code is based on python 3 (version 3.11) and the packages required can be installed with

	python3 -m pip install -r requirements.txt


## Data

This code base focuses on the Rayleigh-Bénard convection (RBC) problem, as a starting point wave equation and darcy flow is solved using the data downloaded from https://zenodo.org/records/10406879 (~2.4GB). It covers instances of the Poisson, Wave, Navier-Stokes, Allen-Cahn, Transport and Compressible Euler equations and Darcy flow. 

Run the script `download_cno_data.py` which downloads all required data into the appropriate folder (it requires 'wget' to be installed on your system).

For RBC problem the data is generated using [Dedalus](https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_2d_rayleigh_benard.html)

See also the [dedalus folder](../dedalus/)


## Model Training

To run on JUWELS booster use [submit_juwels.sbatch.sh](./submit_juwels.sbatch.sh) file that activates the python virtual environment and loads the required modules using [setup.sh](../setup.sh) and executes the  [train_FNO.py](./train_FNO.py) file.

[DarcyWave.ipynb](./DarcyWave.ipynb) and [RBC.ipynb](./RBC.ipynb) jupyter notebook is also available.


## Note

Training can be done using CPU or a single GPU.

GPU testing done on JUWELS Booster: 1X NVIDIA A100 (40GB) GPU with:
- GCC-12.3.0 
- OpenMPI-4.1.5 
- CUDA-12 

The following files correspond to:


- [DarcyWave.ipynb](./DarcyWave.ipynb): Jupyter notebook for solving Wave and Darcy equations using FNO
- [RBC.ipynb](./RBC.ipynb): Jupyter notebook for solving RBC2D equation using FNO
- [download_cno_data.py](./download_cno_data.py): Data for solving Wave and Darcy equations
- [problems.py](./problems.py): Dataloader for FNO model
- [fourier_operator.py](./fourier_operator.py): FNO modules
- [utils.py](./utils.py): Supplemment functions
- [DataVisual.ipynb](./DataVisual.ipynb): Data preprocessing
- [train_FNO.py](./train_FNO.py): FNO model training 
- [inference.py](./inference.py): FNO model inference
- [submit_juwels.sbatch.sh](./submit_juwels.sbatch.sh): Example sbatch script for training FNO in JUWELS booster



## Reference:
- [Convolutional Operators Code](https://github.com/bogdanraonic3/ConvolutionalNeuralOperator)
- [Neural Operators code](https://github.com/neuraloperator/neuraloperator)
