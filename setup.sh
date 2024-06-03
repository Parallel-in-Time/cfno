#!/bin/bash

#load modules
module purge 
module load Stages/2024 
module load GCC/12.3.0 
module load Python/3.11.3 
module load OpenMPI/4.1.5 \
            CUDA/12 \
            FFTW/3.3.10 \
            HDF5/1.14.2 \
            SciPy-Stack/2023a \
            mpi4py/3.1.4   \
            FFTW.MPI     \
            PyTorch/2.1.2 \
            h5py/3.9.0-serial  \
            Pillow-SIMD/9.5.0  \
            tqdm/4.66.1  \
            
# module list
    
# Activate your Python virtual environment
source /p/project/cexalab/john2/NeuralOperators/py_venv/bin/activate
    
# Ensure python packages installed in the virtual environment are always prefered
export PYTHONPATH=/p/project/cexalab/john2/NeuralOperators/py_venv/lib/python*/site-packages:${PYTHONPATH}