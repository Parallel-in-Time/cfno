# Code to generate RBC data with Dedalus

:scroll: _Recommended installation approach_ => [build from source](https://dedalus-project.readthedocs.io/en/latest/pages/installation.html#building-from-source)

## Utility libraries

- [simu.py](./simu.py) : utility module containing functions and classes to generate data with Dedalus.
- [conRuns.py](./convRuns.py) : script used to determine critical Rayleigh for one given space grid
- [plotContours.py](./plotContours.py) : plotting script from dedalus
- [post.py](./post.py) : script for quick post-processing (spectrum and profile extraction)

## Generate data

Only the `simu.py` module is needed. Data generation can be executed with those two lines :

```python
from simu import runSimu, generateChunkPairs

dirName = "dataset"
resFactor = 1   # use base resolution (256,64)
Rayleigh = 1e7  # Rayleigh number below critical 

# run simulation and generate output data every 0.1 sec
# -> may take some time, can be run on a separate script
runSimu(dirName, Rayleigh, resFactor)   

N = 10
M = 4
pairs = generateChunkPairs(dirName, N, M)  
# pairs of chunks of size M, covering N Delta_t = 1sec
# -> chunk shape : (M, 3, Nx, Nz)
```

The `generateChunkPairs` can be also used to generate chunks with smaller grid size, _e.g_ :

```python
pairs = generateChunkPairs(dirName, N, M, xStep=2, zStep=2)  
# -> chunk shape : (M, 3, Nx//2, Nz//2)
```

with a similar idea using a `tStep` parameter 
and can shuffle the pairs provided a given `shuffleSeed` parameter (default to None -> no shuffle).