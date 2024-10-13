# Base scripts for the FNO pipeline

1. [generateData.py](./generateData.py) : run Dedalus simulation to generate data (can be run in parallel with MPI)
2. [createDataset.py](./createDataset.py) : process simulation data into a dataset that can be used for training
3. [viewDataset.py](./viewDataset.py) : visualize samples (input/output) from a given dataset 

All script have integrated command line documentation available with `./scriptName.py -h` (or `--help`), which show the default value for all parameters.
