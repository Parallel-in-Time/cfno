#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys, os
sys.path.insert(2, os.getcwd())
import numpy as np
from cfno.data.preprocessing import HDF5Dataset
from cfno.training.pySDC import FourierNeuralOp
from cfno.simulation.post import contourPlot
from cfno.utils import readConfig

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
    "--iXBeg", default=0, help="xPatch start index", type=int)
parser.add_argument(
    "--iYBeg", default=0, help="yPatch start index", type=int)
parser.add_argument(
    "--iXEnd", default=256, help="xPatch end index", type=int)
parser.add_argument(
    "--iYEnd", default=64, help="yPatch end index", type=int)
parser.add_argument(
    "--outType", default="solution", help="type of output", choices=["solution", "update"])
parser.add_argument(
    "--refScales", action="store_true", help="use the same scales as the reference field")
parser.add_argument(
    "--saveFig", default="modelView", help="output name for contour figure")
parser.add_argument(
    "--config", default="config.yaml", help="configuration file")
parser.add_argument(
    "--get_subdomain_output", action="store_true", help="Get subdomain output")
parser.add_argument(
    "--use_full_input", action="store_true", help="Use full input")
args = parser.parse_args()

dataFile = args.dataFile
checkpoint = args.checkpoint
var = args.var
iSample = args.iSample
outType = args.outType
refScales = args.refScales
saveFig = args.saveFig

# Need model config to get output of shape different to input
# .i.e when using args.get_subdomain_ouput
if args.get_subdomain_output:
    if args.config is not None:
        config = readConfig(args.config)
        args.__dict__.update(**config["model"])
        modelConfig = dict(config.model)
    else:
        raise ValueError("Model configuration is required for FNO to output subdomain.")

iXBeg = args.iXBeg
iYBeg = args.iYBeg
iXEnd = args.iXEnd
iYEnd = args.iYEnd
print(f'Args: {args}')
# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
dataset = HDF5Dataset(dataFile)

nSamples = len(dataset)
assert iSample < nSamples, f"iSample={iSample} to big for {nSamples} samples"

xGrid, yGrid = dataset.grid
u0, uRef_full = dataset.sample(iSample)
print(f'Shape of u0: {u0.shape}, uRef: {uRef_full.shape}')

if args.use_full_input:
    uInit = u0[varChoices.index(var)]
    input = u0
    if args.get_subdomain_output:
        uRef = uRef_full[varChoices.index(var), iXBeg:iXEnd, iYBeg:iYEnd].copy()
    else:
        uRef = uRef_full[varChoices.index(var)].copy()
        iXBeg = 0
        iYBeg = 0
        iXEnd = input.shape[1]
        iYEnd = input.shape[2]
else:
    uInit = u0[varChoices.index(var), iXBeg:iXEnd, iYBeg:iYEnd]
    input = u0[:,iXBeg:iXEnd, iYBeg:iYEnd]
    uRef = uRef_full[varChoices.index(var), iXBeg:iXEnd, iYBeg:iYEnd].copy()

if args.get_subdomain_output:
    model = FourierNeuralOp(model=modelConfig, checkpoint=checkpoint)
else:
    model = FourierNeuralOp(checkpoint=checkpoint)

uPred = model(input)[varChoices.index(var)].copy()
print(f'Shape of uInit: {uInit.shape}, uRef:{uRef.shape}, uPred: {uPred.shape}')

# xExpandedGrid = np.linspace(iXBeg, iXEnd, uPred.shape[0] + 1)
# yExpandedGrid = np.linspace(iYBeg, iYEnd, uPred.shape[1] + 1)
# # Create 2D grids for pcolormesh
# X, Y = np.meshgrid(xExpandedGrid, yExpandedGrid, indexing='ij')
Y, X = xGrid[iXBeg:iXEnd], yGrid[iYBeg:iYEnd]

if dataset.outType == "update":
    uRef /= dataset.outScaling

if uRef.shape != uInit.shape:
    padded_uRef = np.zeros_like(uInit)
    padded_uPred = np.zeros_like(uInit)
    padded_uRef[iXBeg:iXEnd, iYBeg:iYEnd] = uRef[:,:].copy()
    padded_uPred[iXBeg:iXEnd, iYBeg:iYEnd] = uPred[:,:].copy()
    print(f'padded uRef: {padded_uRef.shape}, uPred: {padded_uPred.shape}')          
    if outType == "solution" and dataset.outType == "update":
        padded_uRef += uInit
    if outType == "update" and dataset.outType == "solution":
        padded_uRef -= uInit
    if outType == "update":
        padded_uPred -= uInit
    uPred[:,:] = padded_uPred[iXBeg:iXEnd, iYBeg:iYEnd]
    uRef[:,:] = padded_uRef[iXBeg:iXEnd, iYBeg:iYEnd]
else:
    if outType == "solution" and dataset.outType == "update":
        uRef += uInit
    if outType == "update" and dataset.outType == "solution":
        uRef -= uInit
    if outType == "update":
       uPred -= uInit

contourPlot(
    uPred, X, Y, title=f"Model {outType} for {var} using sample {iSample}",
    refField=uRef, refTitle=f"Dedalus reference (dt={dataset.infos['dtInput'][()]:1.2g}s)",
    saveFig=f'{saveFig}_{outType}.jpg', closeFig=False, refScales=refScales)
print(f" -- saved {var} contour for sample {iSample}")
contourPlot(
    np.abs(uPred-uRef), X, Y, title=f"Model {outType} error for {var} using sample {iSample}\nDedalus reference (dt={dataset.infos['dtInput'][()]:1.2g}s)",
    refField=None, refTitle=None,
    saveFig=f'{saveFig}_{outType}_error.jpg', closeFig=False, refScales=False)
print(f" -- saved {var} contour for sample {iSample}")