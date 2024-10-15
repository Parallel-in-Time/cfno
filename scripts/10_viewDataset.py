#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import h5py
import argparse

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
    "--saveFig", default="sampleView.jpg", help="output name for contour figure")
args = parser.parse_args()

dataFile = args.dataFile
var = args.var
iSample = args.iSample
saveFig = args.saveFig

# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
data = h5py.File(dataFile, "r")
nSamples = len(data["inputs"])
assert iSample < nSamples, f"iSample={iSample} to big for {nSamples} samples"
xGrid, yGrid = data["infos/xGrid"][:], data["infos/yGrid"][:]
print(f"Reading {dataFile} ...")
print(f" -- inSize : {data['infos/inSize'][()]}")
print(f" -- outStep : {data['infos/outStep'][()]}")
print(f" -- inStep : {data['infos/inStep'][()]}")
print(f" -- grid shape : ({xGrid.size}, {yGrid.size})")
print(f" -- grid domain : [{xGrid.min():.1f}, {xGrid.max():.1f}] x [{yGrid.min():.1f}, {yGrid.max():.1f}]")
print(f" -- dtData : {data['infos/dtData'][()]:1.2g}")
print(f" -- dtInput : {data['infos/dtInput'][()]:1.2g}")
print(f" -- outType : {data['infos/outType'][()].decode('utf-8')}")
print(f" -- outScaling : {data['infos/outScaling'][()]:1.2g}")
contourPlot(
    data["inputs"][iSample, varChoices.index(var)].T,
    xGrid, yGrid, title=f"Input for {var}",
    refField=data["outputs"][iSample, varChoices.index(var)].T,
    refTitle=f"Output for {var} after dt={data['infos/dtInput'][()]:1.2g}s",
    saveFig=saveFig, closeFig=False)
print(f" -- saved {var} contour for sample {iSample}")
