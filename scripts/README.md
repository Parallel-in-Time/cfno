# Base scripts for the FNO pipeline

## Quick description

Can be summarized in 4 main stages :

1. **Simulation** : generate simulation data with Dedalus, [`01_simu.py` script](./01_simu.py).
2. **Sampling** : create a training dataset by sampling Dedalus simulation data, [`02_sample.py` script](./02_sample.py).
3. **Training** : train a FNO model on a given dataset, [`03_train.py` script](./03_train.py)
4. **Evaluation** : evaluate the trained FNO with some metrics, [`04_eval.py` script](./04_eval.py)
5. **SDC Run** : run SDC with FNO initialization, [`05_runSDC.py` script](./05_runSDC.py)

Each script can be run separately with command-line arguments, with **all arguments having default value** (use `-h` for more details), ex :

```bash
$ ./01_simu.py -h
```

but also using a `config.yaml` file storing all scripts argument, that will overwrite any argument
passed in command line, if provided in the config file, ex :

```bash
$ ./01_simu.py --config pathToConfig.yaml
```

> :warning: Per default, all pipeline scripts don't use a config file if not provided, except for [`03_train.py`](./03_train.py)
> that requires one, and if not provided will use a default `config.yaml` provided in the current directory 
> (see the base [`config.yaml`](./config.yaml) for the pipeline default configuration).

In fact, you can run the all pipeline using default value (from the scripts and the base [`config.yaml`] file), in order to see if everything works fine ...

```bash
$ ./01_simu.py      # this may take a very long time, better to run with MPI
$ ./02_sample.py    # run once, can be used for many training ...
$ ./03_train.py     # faster when run on GPU 
$ ./04_eval.py      # can be run separately from training ...
```

There is also some companion scripts that can be used in parallel to the pipeline scripts (they don't require any config file):

- [10_viewDataset.py](./10_viewDataset.py) : print infos from a dataset, and can plot some contours of its inputs / outputs
- [11_modelOutput.py](./11_modelOutput.py) : plot solution (or update) contours of a model on a given sample of a dataset
- [12_inspectModel.py](./12_inspectModel.py) : print model configuration and status from a checkpoint file
- [13_plotLoss.py](./13_plotLoss.py) : plot loss evolution from a file storing it

> :scroll: Examples of slurm scripts using those scripts are provided in the [`slurm`](./slurm/) folder ...


## Stage 1 : simulation

Use the [`01_simu.py` script](./01_simu.py) to run `nSimu` simulation, that will start accumulate data after `tInit` seconds,
and write a solution field every `dtData` seconds, until `tEnd` seconds are done. 
All simulation file will be stored in a `dataDir` folder.

> :warning: `tEnd` doesn't take into account `tInit`, so the total simulation time (initial run + data accumulation) is `tInit` + `tEnd`.

This script can be run in parallel using MPI, and can use a the arguments provided in a `simu` section of a config file.


## Stage 2 : sampling

Use the [`02_sample.py` script](./02_sample.py) to create a dataset stored in `dataFile`, from simulation data stored in `dataDir`.
It uses three main parameters for sampling :

- `inSize` : number of time-step contained in one input (only `inSize=1` implemented for now)
- `outStep` : number of `dtData` between an input and its output (i.e the time-step size of the update)
- `inStep` : the number of `dtData` to jump before taking the next input (and associated output ...)

In addition, there is two additional parameters that can be used

- `outType` : 
    - if `"solution"`, then each output is built by simply taking the time-stepper solution
    - if `"update"`, then each output is built by taking the time-stepper update, multiply by a given scaling factor
- `outScaling` : if `outType="update"`, the scaling factor used to build the output

If a config file is given, then it will use any parameters provided in a `simu`, `sample` and `data` section, 
see the base [`config.yaml`](./config.yaml) for reference ... 


## Stage 3 : training

Use the [`03_train.py` script](./03_train.py) to train a model using a training dataset stored in `dataFile` (section `data` in the config file).
Most of the training settings have to be specified in a config file, see the base [`config.yaml`](./config.yaml) for reference ... 

It will run on GPU is one is available, else on CPU. Specifying `seed: null` will simply separate the data using the `trainRatio`
without shuffling the data. For instance, if 10 simulations were run to generate data, `trainRatio=0.8` takes the data of 8 simulations
for training, and the data of the 2 last simulations for validation.


## Stage 4 : evaluation

Use the [`04_eval.py` script](./04_eval.py) to evaluate a given model stored in a `checkpoint` file,
on one simulation of a dataset stored in `dataFile`. 

> :mega: This script don't necessarily require a config file : the whole model can be instantiated
> with the `checkpoint` file, since all model settings are stored in there, and are sufficient
> for inference only. 

Evaluation metrics are (for now) :

- averaged spectrum for $u_x$ and $u_z$
