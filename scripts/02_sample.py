#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import h5py
import argparse
import numpy as np

from fnop.simulation.post import OutputFiles
from fnop.utils import readConfig

# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Create training dataset from Dedalus simulation data',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--dataDir", default="simuData", help="directory containing simulation data")
parser.add_argument(
    "--inSize", default=1, help="input size", type=int)
parser.add_argument(
    "--outStep", default=1, help="output step", type=int)
parser.add_argument(
    "--inStep", default=5, help="input step", type=int)
parser.add_argument(
    "--outType", default="solution", help="output type in the dataset",
    choices=["solution", "update"])
parser.add_argument(
    "--outScaling", default=1, type=float, help="scaling factor for the output (ignored with outType=solution !)")
parser.add_argument(
    "--dataFile", default="dataset.h5", help="name of the dataset HDF5 file")
parser.add_argument(
    "--config", default=None, help="config file, overwriting all parameters specified in it")
args = parser.parse_args()

if args.config is not None:
    config = readConfig(args.config)
    assert "sample" in config, f"config file needs a data section"
    args.__dict__.update(**config.data)
    if "simu" in config and "dataDir" in config.simu:
        args.dataDir = config.simu.dataDir
    if "data" in config:
        for key in ["outType", "outScaling", "dataFile"]:
            if key in config.data: args.__dict__[key] = config.data[key]

dataDir = args.dataDir
inSize = args.inSize
outStep = args.outStep
inStep = args.inStep
outType = args.outType
outScaling = args.outScaling
dataFile = args.dataFile

# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
assert inSize == 1, "inSize != 1 not implemented yet ..."
simDirs = glob.glob(f"{dataDir}/simu_*")

# -- retrieve informations from first simulation
outFiles = OutputFiles(f"{simDirs[0]}/run_data")

times = outFiles.times().ravel()
dtData = times[1]-times[0]
dtInput = dtData*outStep
xGrid, yGrid = outFiles.x, outFiles.y

nFields = sum(outFiles.nFields)
sRange = range(0, nFields-inSize-outStep+1, inStep)
nSamples = len(sRange)

print(f"Creating dataset from {len(simDirs)} simulations, {nSamples} samples each ...")
dataset = h5py.File(dataFile, "w")
for name in ["inSize", "outStep", "inStep", "outType", "outScaling",
             "dtData", "dtInput", "xGrid", "yGrid"]:
    try:
        dataset.create_dataset(f"infos/{name}", data=np.asarray(eval(name)))
    except:
        dataset.create_dataset(f"infos/{name}", data=eval(name))

dataShape = (nSamples*len(simDirs), *outFiles.shape)
inputs = dataset.create_dataset("inputs", dataShape)
outputs = dataset.create_dataset("outputs", dataShape)
for iSim, dataDir in enumerate(simDirs):
    outFiles = OutputFiles(f"{dataDir}/run_data")
    print(f" -- sampling data from {outFiles.folder}")
    for iSample, iField in enumerate(sRange):
        inpt, outp = outFiles.fields(iField), outFiles.fields(iField+outStep)
        if outType == "update":
            outp -= inpt
            if outScaling != 1:
                outp *= outScaling
        inputs[iSim*nSamples + iSample] = inpt
        outputs[iSim*nSamples + iSample] = outp
print(" -- done !")
