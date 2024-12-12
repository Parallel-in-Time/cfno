#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base script to evaluate FNO on distributed space solutions
"""
import numpy as np

from cfno.data.preprocessing import HDF5Dataset
from cfno.training.pySDC import FourierNeuralOp
from cfno.simulation.post import contourPlot

varChoices = ["vx", "vz", "b", "p"]
outType = "update"
var = "b"
iSample = 1900
saveFig = False
refScales = True
nDomains = 1

dataset = HDF5Dataset("../dataset_dt1e-3.h5")

model = FourierNeuralOp(checkpoint="../training_dt1e-3_update/model.pt")

xGrid, yGrid = dataset.grid
u0, uRef = dataset.sample(iSample)

uInit = u0[varChoices.index(var)].T
uRef = uRef[varChoices.index(var)].T.copy()


nZLoc = yGrid.size // nDomains
uPred = np.zeros_like(u0)
for i in range(nDomains):
    u0Loc = u0[:, :, i*nZLoc:(i+1)*nZLoc]
    uPred[:, :, i*nZLoc:(i+1)*nZLoc] = model(u0Loc)
uPred = uPred[varChoices.index(var)].T.copy()

if dataset.outType == "update":
    uRef /= dataset.outScaling
if outType == "solution" and dataset.outType == "update":
    uRef += uInit
if outType == "update" and dataset.outType == "solution":
    uRef -= uInit

if outType == "update":
    uPred -= uInit

contourPlot(
    uPred, xGrid, yGrid, title=f"Model {outType} for {var} using sample {iSample}",
    refField=uRef, refTitle=f"Dedalus reference (dt={dataset.infos['dtInput'][()]:1.2g}s)",
    saveFig=saveFig, closeFig=False, refScales=refScales)


uPredCoarse = model(u0[:, :, 1::2])[varChoices.index(var)].T.copy()
if outType == "update":
    uPredCoarse -= uInit[1::2]
uRefCoarse = uRef[1::2]

contourPlot(
    uPredCoarse, xGrid, yGrid[1::2], title=f"Coarse model {outType} for {var} using sample {iSample}",
    refField=uRefCoarse, refTitle=f"Dedalus reference (dt={dataset.infos['dtInput'][()]:1.2g}s)",
    saveFig=saveFig, closeFig=False, refScales=refScales)
