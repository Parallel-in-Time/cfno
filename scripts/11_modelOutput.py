#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
import os
sys.path.insert(2, os.getcwd())
from cfno.data.preprocessing import HDF5Dataset
from cfno.training.pySDC import FourierNeuralOp
from cfno.simulation.post import contourPlot

varChoices = ["vx", "vz", "b", "p"]

# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='View model output from inputs stored in a HDF5 dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--dataFile", default="dataset.h5", help="name of the dataset HDF5 file")
parser.add_argument(
    "--checkpoint", default="model.pt", help="name of the model checkpoint file")
parser.add_argument(
    "--var", default="b", help="variable to view", choices=varChoices)
parser.add_argument(
    "--iSample", default=0, help="sample index", type=int)
parser.add_argument(
    "--outType", default="solution", help="type of output", choices=["solution", "update"])
parser.add_argument(
    "--refScales", action="store_true", help="use the same scales as the reference field")
parser.add_argument(
    "--saveFig", default="modelView.jpg", help="output name for contour figure")
args = parser.parse_args()

dataFile = args.dataFile
checkpoint = args.checkpoint
var = args.var
iSample = args.iSample
outType = args.outType
refScales = args.refScales
saveFig = args.saveFig

# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
dataset = HDF5Dataset(dataFile)

nSamples = len(dataset)
assert iSample < nSamples, f"iSample={iSample} to big for {nSamples} samples"

xGrid, yGrid = dataset.grid
u0, uRef = dataset.sample(iSample)

uInit = u0[varChoices.index(var)].T
uRef = uRef[varChoices.index(var)].T.copy()

model = FourierNeuralOp(checkpoint=checkpoint)
uPred = model(u0)[varChoices.index(var)].T.copy()

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
print(f" -- saved {var} contour for sample {iSample}")
