#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

import matplotlib.pyplot as plt

from fnop.utils import readConfig
from fnop.data import HDF5Dataset
from fnop.fno import FourierNeuralOp
from fnop.simulation.post import computeMeanSpectrum, getModes


# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Evaluate a model on a given dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--dataFile", default="dataset.h5", help="name of the dataset HDF5 file")
parser.add_argument(
    "--checkpoint", default="model.pt", help="name of the file storing the model")
parser.add_argument(
    "--iSimu", default=8, type=int, help="index of the simulation to eval with")
parser.add_argument(
    "--imgExt", default="png", help="extension for figure files")
parser.add_argument(
    "--config", default=None, help="configuration file")
args = parser.parse_args()

if args.config is not None:
    config = readConfig(args.config)
    if "eval" in config:
        args.__dict__.update(**config["eval"])
    if "data" in config and "dataFile" in config["data"]:
        args.dataFile = config.data.dataFile
    if "train" in config and "checkpoint" in config["train"]:
        args.checkpoint = config.train.checkpoint
        if "trainDir" in config.train:
            FourierNeuralOp.TRAIN_DIR = config.train.trainDir

dataFile = args.dataFile
checkpoint = args.checkpoint
iSimu = args.iSimu
imgExt = args.imgExt

# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
dataset = HDF5Dataset(dataFile)
model = FourierNeuralOp(checkpoint=checkpoint)

nSamples = dataset.infos["nSamples"][()]
nSimu = dataset.infos["nSimu"][()]
assert iSimu < nSimu, f"cannot evaluate with iSimu={iSimu} with only {nSimu} simu"
indices = slice(iSimu*nSamples, (iSimu+1)*nSamples)

u0Values = dataset.inputs[indices]
uRefValues = dataset.outputs[indices]
uPredValues = model(u0Values)

sxRef, szRef = computeMeanSpectrum(uRefValues)
sxPred, szPred = computeMeanSpectrum(uPredValues)
k = getModes(dataset.grid[0])

plt.figure()
p = plt.loglog(k, sxRef.mean(axis=0), '--', label="sx (ref)")
plt.loglog(k, sxPred.mean(axis=0), c=p[0].get_color(), label="sx (model)")

p = plt.loglog(k, szRef.mean(axis=0), '--', label="sz (ref)")
plt.loglog(k, szPred.mean(axis=0), c=p[0].get_color(), label="sz (model)")

plt.legend()
plt.grid()
plt.ylabel("spectrum")
plt.xlabel("wavenumber")
plt.ylim(bottom=1e-10)
plt.tight_layout()
plt.savefig(f"spectrum.{imgExt}")

plt.xlim(left=50)
plt.ylim(top=1e-5)
plt.savefig(f"spectrum_HF.{imgExt}")
