#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import argparse

from fnop.data import HDF5Dataset
from fnop.simulation.post import contourPlot

varChoices = ["vx", "vz", "b", "p"]

# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='View sample fields within a dataset stored in a HDF5 file',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--dataFile", default="dataset.h5", help="name of the dataset HDF5 file")
parser.add_argument(
    "--var", default="b", help="variable to view", choices=varChoices)
parser.add_argument(
    "--iSample", default=0, help="sample index", type=int)
parser.add_argument(
    "--outType", default="solution", help="type of output", choices=["solution", "update"])
parser.add_argument(
    "--saveFig", default="sampleView.jpg", help="output name for contour figure (empty to only print infos)")
args = parser.parse_args()

dataFile = args.dataFile
var = args.var
iSample = args.iSample
outType = args.outType
saveFig = args.saveFig

# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
dataset = HDF5Dataset(dataFile)

nSamples = len(dataset)
assert iSample < nSamples, f"iSample={iSample} to big for {nSamples} samples"

print(f"Reading {dataFile} ...")
dataset.printInfos()

if not saveFig:
    sys.exit()

xGrid, yGrid = dataset.grid
u0, u1 = dataset.sample(iSample)

u0 = u0[varChoices.index(var)].T
u1 = u1[varChoices.index(var)].T

if dataset.outType == "update":
    u1 /= dataset.outScaling
if outType == "solution" and dataset.outType == "update":
    u1 += u0
if outType == "update" and dataset.outType == "solution":
    u1 -= u0
outName = "Solution" if outType == "solution" else "Update"

contourPlot(
    u0, xGrid, yGrid, title=f"Input for {var} in sample {iSample}",
    refField=u1, refTitle=f"{outName} after dt={dataset.infos['dtInput'][()]:1.2g}s",
    saveFig=saveFig, closeFig=False)
print(f" -- saved {var} contour for sample {iSample}")
