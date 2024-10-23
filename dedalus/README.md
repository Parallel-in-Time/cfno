# Code to generate RBC data with Dedalus

:ghost: DEPRECATED ...

:scroll: _Recommended installation approach for `dedalus`_ => [build from source](https://dedalus-project.readthedocs.io/en/latest/pages/installation.html#building-from-source)

## Utility libraries (:ghost: deprecated ...)

- [rbc_simulation.py](./rbc_simulation.py) : utility module containing function to generate data with Dedalus.
- [convRuns.py](./convRuns.py) : script used to determine critical Rayleigh for one given space grid
- [plotContours.py](./plotContours.py) : plotting script from dedalus
- [data_processing.py](./data_processing.py) : script for quick post-processing (data, spectrum and profile extraction)

## Generate data

Data generation can be executed as:

```python
from rbc_simulation import runSim
from data_processing import generateChunkPairs

dirName = "dataset"
resFactor = 1   # use base resolution (256,64)
Rayleigh = 1e7  # Rayleigh number below critical 

# run simulation and generate output data every 0.1 sec
# -> may take some time, can be run on a separate script
runSim(dirName, Rayleigh, resFactor)   

N = 10
M = 4
pairs = generateChunkPairs(dirName, N, M) 
# pairs shape: (nTimes-M-N+1, 2, M, 4, Nx, Nz) 
# pairs of chunks of size M, covering N Delta_t = 10 * 0.1 sec = 1sec
# -> chunk shape : (M, 4, Nx, Nz)
```

The `generateChunkPairs` can be also used to generate chunks with smaller grid size, _e.g_ :

```python
pairs = generateChunkPairs(dirName, N, M, xStep=2, zStep=2)  
# -> chunk shape : (M, 4, Nx//xStep, Nz//zStep)
```

with a similar idea using a `tStep` parameter 
and can shuffle the pairs provided a given `shuffleSeed` parameter (default to None -> no shuffle).