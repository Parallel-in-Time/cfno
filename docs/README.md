# Some docs on the FNO stuff ...

## Generated

- [fno.svg](./fno.svg) : graphical representation of a 1D FNO with one Fourier layer
- [fno2d.svg](./fno2d.svg) : graphical representation of a 2D FNO with one Fourier layer
- [cfno2d.svg](./cfno2d.svg): graphical representation of a 2D Chebyshev-Fourier Neural Operator (CFNO) with one Fourier layer 

## Useful links

- [base FNO paper](https://arxiv.org/pdf/2010.08895)
- [Github FNO package](https://github.com/neuraloperator/neuraloperator)
- [DCT for torch](https://github.com/zh217/torch-dct)
- [SHNO paper](https://arxiv.org/pdf/2306.03838)
- [SHT for torch](https://github.com/NVIDIA/torch-harmonics)
- [reference code](https://github.com/neuraloperator/neuraloperator.git)

## Code for the original approach (old documentation)

### Model Training

For 2D Rayleigh Benard Convection (RBC) problem the data is generated using [Dedalus](https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_2d_rayleigh_benard.html). See also the [dedalus](./dedalus/) folder.

[Fourier Neural Operator 2D Spatial + Recurrent in time](./fnop/models/fno2d_recurrent.py) and [Fourier Neural Operator 2D Spatial + 1D time](./fnop/models/fno3d.py) solver for RBC 2D.

Example submission script to train on JUWELS Booster is shown in [submit_juwels.sbatch](./launch_scripts/submit_juwels.sbatch.sh).

### Model Inference

Once a model is trained, it is saved into a _checkpoint_ file, that are stored in the [`model_archive` companion repo](https://codebase.helmholtz.cloud/neuralpint/model_archive) as `*.pt` files 
(weights of the model).
Which each of those models is associated a YAML configuration file, that stores all the model setting
parameters.

One given trained model should be used like this :

```python
from fnop.inference import FNOInference

# Load the trained model
model = FNOInference(
    config="path/to/configFile.yaml",
    checkpoint="path/to/checkpointFile.pt")

u0 = ... # some solution of RBC at a given time
# --> numpy.ndarray format, with shape (nVar, nX, nZ), with nVar = 4

# In particular, uInit could be unpacked like this :
vx0, vy0, b0, p0 = u0

# Evaluate the model with a given initial solution using a fixed time-step
u1 = model.predict(u0)  # -> only one time-step !

# --> u1 can also be unpacked like u0, e.g :
vx1, vy1, b1, p1 = u1

# Time-step of the model can be retrieved like this :
dt = model.dt 	# -> can be either an attribute or a property
```