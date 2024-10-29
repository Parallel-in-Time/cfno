#!/bin/bash

#load modules
module purge 
module load Stages/2024 
module load GCC/12.3.0 
module load Python/3.11.3 \
            OpenMPI/4.1.5 \
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
            imageio/2.31.3
            
# module list
if [[ "$SYSTEMNAME" = "juwelsbooster" ]]; then
    echo "********* Using $SYSTEMNAME *********"
    source /p/project1/cexalab/john2/NeuralOperators/py_venv/bin/activate   
    export PYTHONPATH=/p/project1/cexalab/john2/NeuralOperators/py_venv/lib/python*/site-packages:${PYTHONPATH}
elif [[ "$SYSTEMNAME" = "jurecadc" ]]; then 
    echo "********* Using $SYSTEMNAME *********"
    source /p/project1/cexalab/john2/NeuralOperators/no_jureca_env/bin/activate
    export PYTHONPATH=/p/project1/cexalab/john2/NeuralOperators/no_jureca_env/lib/python3.11/site-packages:${PYTHONPATH}
else
    echo "Currently only JURECA-DC and JUWELS Booster are supported"
fi