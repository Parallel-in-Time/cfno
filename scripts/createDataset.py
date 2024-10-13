#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import h5py
import argparse
import numpy as np

from fnop.simulation.post import OutputFiles

# Script parameters
parser = argparse.ArgumentParser(
    description='Create training (and validation) dataset from Dedalus simulation data',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--dataDir", default="generateData", help="dir. containing simulation data")
parser.add_argument(
    "--inSize", default=1, help="input size", type=int)
parser.add_argument(
    "--outStep", default=1, help="output step", type=int)
parser.add_argument(
    "--inStep", default=5, help="input step", type=int)
parser.add_argument(
    "--dataFile", default="dataset.h5", help="name of the dataset HDF5 file")
args = parser.parse_args()

dataDir = args.dataDir
inSize = args.inSize
outStep = args.outStep
inStep = args.inStep
dataFile = args.dataFile

# Script execution
assert inSize == 1, "inSize > 1 not implemented yet ..."
simDirs = glob.glob(f"{dataDir}/simu_*")

# -- retrieve informations from first simulation
outFiles = OutputFiles(f"{simDirs[0]}/run_data")

times = outFiles.times().ravel()
dtData = times[1]-times[0]
dtInput = dtData*outStep
xGrid, zGrid = outFiles.x, outFiles.z

nFields = sum(outFiles.nFields)
sRange = range(0, nFields-inSize-outStep+1, inStep)
nSamples = len(sRange)

print(f"Creating dataset from {len(simDirs)} simulations, {nSamples} samples each ...")
dataset = h5py.File(dataFile, "w")
for name in ["inSize", "outStep", "inStep", "dtData", "dtInput",
             "xGrid", "zGrid"]:
    dataset.create_dataset(f"infos/{name}", data=np.asarray(eval(name)))

dataShape = (nSamples*len(simDirs), *outFiles.shape)
inputs = dataset.create_dataset("inputs", dataShape)
outputs = dataset.create_dataset("outputs", dataShape)
for iSim, simDir in enumerate(simDirs):
    outFiles = OutputFiles(f"{simDir}/run_data")
    print(f" -- sampling data from {outFiles.folder}")
    for iSample, iField in enumerate(sRange):
        inputs[iSim*nSamples + iSample] = outFiles.fields(iField)
        outputs[iSim*nSamples + iSample] = outFiles.fields(iField+outStep)
print(" -- done !")
