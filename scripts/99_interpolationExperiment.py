#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 10:47:07 2025

@author: cpf5546
"""
import numpy as np

from qmat.lagrange import LagrangeApproximation
from qmat.nodes import NodesGenerator

from cfno.data.preprocessing import HDF5Dataset
from cfno.training.pySDC import FourierNeuralOp
from cfno.simulation.post import contourPlot

dataFile = "datasets/dataset_512x128_Ra1e7_dt1e-3_update.h5"
modelFile = "models/model_run24_dt1e-3.pt"


dataset = HDF5Dataset(dataFile)
model = FourierNeuralOp(checkpoint=modelFile)


iSample = 20

u0 = dataset.inputs[iSample]
uRef = dataset.outputs[iSample]

uRef /= dataset.outScaling
uRef += u0

uPred = model(u0)

xGrid = dataset.infos["xGrid"][:]
yGrid = dataset.infos["yGrid"][:]

xGridCoarse = xGrid[::2]
yGridCoarse = NodesGenerator(nodeType="CHEBY-1", quadType="GAUSS").getNodes(yGrid.size//2)
yGridCoarse += 1
yGridCoarse /= 2
R = LagrangeApproximation(
    yGrid, weightComputation="STABLE").getInterpolationMatrix(yGridCoarse)
P = LagrangeApproximation(
    yGridCoarse, weightComputation="STABLE").getInterpolationMatrix(yGrid)


nVar, nX, nZ = u0.shape
# Restriction to coarse grid
u0Coarse = (R @ u0[:, ::2, :].reshape(-1, nZ).T).T.reshape(nVar, nX//2, nZ//2)
uRefCoarse = (R @ uRef[:, ::2, :].reshape(-1, nZ).T).T.reshape(nVar, nX//2, nZ//2)
# Model evaluation
uPredCoarse = model(u0Coarse)
# Interpolation to fine grid
uPred2 = np.fft.irfft(np.fft.rfft(uPredCoarse, axis=1), n=nX, axis=1)
uPred2 = (P @ uPred2.reshape(-1, nZ//2).T).T.reshape(nVar, nX, nZ)

uI = u0[2].T
uM = uPred[2].T
uM2 = uPred2[2].T
uR = uRef[2].T

contourPlot(
    uI, xGrid, yGrid, title="Initial solution",
    refField=uR, refTitle="Dedalus solution reference",
    closeFig=False)

contourPlot(
    uM-uI, xGrid, yGrid, title="Model update using fine initial field",
    refField=uR-uI, refTitle="Dedalus update reference",
    refScales=True, closeFig=False)

contourPlot(
    uM2-uI, xGrid, yGrid, title="Model update using coarse initial field",
    refField=uR-uI, refTitle="Dedalus update reference",
    refScales=True, closeFig=False)
