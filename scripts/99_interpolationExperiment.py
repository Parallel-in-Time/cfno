#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 10:47:07 2025

@author: cpf5546
"""
import numpy as np
import pandas as pd

from qmat.lagrange import LagrangeApproximation
from qmat.nodes import NodesGenerator

from hopi import RectilinearGrid

from cfno.data.preprocessing import HDF5Dataset
from cfno.training.pySDC import FourierNeuralOp
from cfno.simulation.post import contourPlot

# Script parameters
dataFile = "datasets/dataset_512x128_Ra1e7_dt1e-3_update.h5"
modelFile = "models/model_run24_dt1e-3.pt"

iSample = 10        # index of the sample used for evaluation
itpOrder = 9        # interpolation order : odd number of np.inf (Fourier)
injection = True    # use injection for x restriction (if not, use Fourier)
iV = 2              # index of variable to show on contour plots

# Script execution
dataset = HDF5Dataset(dataFile)
model = FourierNeuralOp(checkpoint=modelFile)

u0 = dataset.inputs[iSample]
uRef = dataset.outputs[iSample]

assert dataset.outType == "update"
uRef /= dataset.outScaling
uRef += u0

# Model evaluated on fine solution
uPred = model(u0)

xGrid = dataset.infos["xGrid"][:]
yGrid = dataset.infos["yGrid"][:]

# Model evaluated on coarse solution + interpolation
xGridCoarse = xGrid[::2]
yGridCoarse = NodesGenerator(nodeType="CHEBY-1", quadType="GAUSS").getNodes(yGrid.size//2)
yGridCoarse += 1
yGridCoarse /= 2
R = LagrangeApproximation(
    yGrid, weightComputation="STABLE").getInterpolationMatrix(yGridCoarse)
P = LagrangeApproximation(
    yGridCoarse, weightComputation="STABLE").getInterpolationMatrix(yGrid)
nVar, nX, nZ = u0.shape

# -- restriction to coarse grid
if injection:
    u0Coarse = u0[:, ::2]
    uRefCoarse = uRef[:, ::2]
else:
    u0Coarse = np.fft.irfft(np.fft.rfft(u0, axis=1), n=nX//2, axis=1)/2
    uRefCoarse = np.fft.irfft(np.fft.rfft(uRef, axis=1), n=nX//2, axis=1)/2
u0Coarse = (R @ u0Coarse.reshape(-1, nZ).T).T.reshape(nVar, nX//2, nZ//2)
uRefCoarse = (R @ uRefCoarse.reshape(-1, nZ).T).T.reshape(nVar, nX//2, nZ//2)

# -- evaluation of coarse input
uPredCoarse = model(u0Coarse)

# -- interpolation of coarse output
if itpOrder == np.inf:
    uPred2 = np.fft.irfft(np.fft.rfft(uPredCoarse, axis=1), n=nX, axis=1)*2
    uPred2 = (P @ uPred2.reshape(-1, nZ//2).T).T.reshape(nVar, nX, nZ)
else:
    RectilinearGrid.VERBOSE = False
    grid = RectilinearGrid(
        itpOrder,
        xG=[xGridCoarse, yGridCoarse], xF=[xGrid, yGrid],
        boundary=["PER", "WALL"], xL=[0, 0], xR=[4, 1])
    uPred2 = np.empty_like(uPred)
    uPred2[0] = grid.interpolate(uPredCoarse[0], lVal=0, rVal=0)
    uPred2[1] = grid.interpolate(uPredCoarse[1], lVal=0, rVal=0)
    uPred2[2] = grid.interpolate(uPredCoarse[2], lVal=1, rVal=0)
    uPred2[3] = grid.interpolate(uPredCoarse[3], lVal=0, rVal=0)

def norm(x):
    return np.linalg.norm(x, axis=(-2, -1))

def computeError(uPred, uRef):
    diff = norm(uPred-uRef)
    nPred = norm(uPred)
    return diff/nPred


errors = pd.DataFrame(
    data={
        "model on fine": computeError(uPred, uRef),
        "model on coarse": computeError(uPredCoarse, uRefCoarse),
        f"model on coarse + inter. [{itpOrder}]": computeError(uPred2, uRef)
        },
    index=["u_x", "u_z", "b", "p"]).T
print(errors.to_markdown())

# Contour plot
contourPlot(
    u0[2].T, xGrid, yGrid,
    title="Initial solution",
    refField=uRef[2].T, refTitle="Dedalus solution reference",
    saveFig="solution.png", closeFig=True)

contourPlot(
    uPred[iV].T - u0[iV].T, xGrid, yGrid,
    title="Model update on fine input",
    refField=uRef[iV].T - u0[iV].T,
    refTitle="Dedalus update reference",
    saveFig="update-fine.png", refScales=True, closeFig=True)

contourPlot(
    uPredCoarse[iV].T - u0Coarse[iV].T, xGridCoarse, yGridCoarse,
    title="Model update using coarse initial field (coarse)",
    refField=uRefCoarse[iV].T - u0Coarse[iV].T,
    refTitle="Dedalus update reference (coarse)",
    saveFig="update-coarse.png", refScales=True, closeFig=True)

contourPlot(
    uPred2[iV].T-u0[iV].T, xGrid, yGrid,
    title=f"Model update on coarse input + interpolation [{itpOrder}]",
    refField=uRef[iV].T-u0[iV].T, refTitle="Dedalus update reference",
    saveFig="update-coarse-inter.png", refScales=True, closeFig=True)
