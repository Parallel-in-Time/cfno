#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import h5py
import argparse

from fnop.simulation.post import contourPlot

varChoices = ["vx", "vz", "b", "p"]

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
    "--saveFig", default="sampleView.jpg", help="output name for contour figure")
args = parser.parse_args()

dataFile = args.dataFile
var = args.var
iSample = args.iSample
saveFig = args.saveFig

# Script execution
data = h5py.File(dataFile, "r")
nSamples = len(data["inputs"])
assert iSample < nSamples, "iSample={iSample} to big for {nSamples} samples"
xGrid, zGrid = data["infos/xGrid"][:], data["infos/zGrid"][:]
contourPlot(
    data["inputs"][iSample, varChoices.index(var)].T,
    xGrid, zGrid, title=f"Input for {var}",
    refField=data["outputs"][iSample, varChoices.index(var)].T,
    refTitle=f"Output for {var}",
    saveFig=saveFig, closeFig=False)
