#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
sys.path.insert(2, os.getcwd())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cfno.utils import readConfig
from cfno.data.preprocessing import HDF5Dataset
from cfno.training.pySDC import FourierNeuralOp
from cfno.simulation.post import computeMeanSpectrum, getModes, contourPlot


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
    "--evalDir", default="eval", help="directory to store the evaluation results")
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
evalDir = args.evalDir

HEADER = """
# FNO evaluation on validation dataset

- simulation index: {iSimu}
- model name: {checkpoint}
- dataset : {dataFile}
    - nSamples : {nSamples}
    - dtInput (between input and output of the model) : {dtInput}
    - dtSample (between two samples) : {dtSample}
    - outType : {outType}
    - outScaling : {outScaling}

"""
op = os.path
with open(op.dirname(op.abspath(op.realpath(__file__)))+"/04_eval_template.md") as f:
    TEMPLATE = f.read()

def sliceToStr(s:slice):
    out = ":"
    if s.start is not None:
        out = str(s.start)+out
    if s.stop is not None:
        out = out+str(s.stop)
    return out

# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
dataset = HDF5Dataset(dataFile)
model = FourierNeuralOp(checkpoint=checkpoint)
os.makedirs(evalDir, exist_ok=True)

nSamples = dataset.infos["nSamples"][()]
print(f'nSamples: {nSamples}')
nSimu = dataset.infos["nSimu"][()]
assert iSimu < nSimu, f"cannot evaluate with iSimu={iSimu} with only {nSimu} simu"
indices = slice(iSimu*nSamples, (iSimu+1)*nSamples)

# Initial solution for all samples
u0 = dataset.inputs[indices]

# Reference solution for all samples
uRef = dataset.outputs[indices].copy()
if dataset.outType == "update":
    uRef /= dataset.outScaling
    uRef += u0

# Create summary file, and write header
def fmt(hdfFloat): return float(hdfFloat[()])

summary = open(f"{evalDir}/eval.md", "w")
summary.write(HEADER.format(
    iSimu=iSimu, checkpoint=checkpoint, dataFile=dataFile, nSamples=nSamples,
    dtInput=fmt(dataset.infos["dtInput"]), dtSample=fmt(dataset.infos["dtSample"]),
    outType=dataset.outType, outScaling=dataset.outScaling))

decomps = [
    [(slice(None), slice(None))],   # full domain evaluation

    [(slice(0, 64), slice(None)),   # 4 domains distributed in X direction
     (slice(64, 128), slice(None)),
     (slice(128, 192), slice(None)),
     (slice(192, 256), slice(None))],

    [(slice(None), slice(0, 16)),   # 4 domains distributed in Z direction
     (slice(None), slice(16, 32)),
     (slice(None), slice(32, 48)),
     (slice(None), slice(48, 64))],
    ]


for iDec in range(len(decomps)):
    slices = decomps[iDec]

    uPred = np.zeros_like(uRef)
    print(f"Computing One-Step prediction for D{iDec}")
    for j, s in enumerate(slices):
        print(f" -- slice {j+1}/{len(slices)}")
        _ = slice(None)
        uPred[(_, _, *s)] = model(u0[(_, _, *s)])
    print(" -- done !")

    # -------------------------------------------------------------------------
    # -- Relative error over time
    # -------------------------------------------------------------------------
    def norm(x):
        return np.linalg.norm(x, axis=(-2, -1))

    def computeError(uPred, uRef):
        diff = norm(uPred-uRef)
        nPred = norm(uPred)
        return diff/nPred

    err = computeError(uPred, uRef)
    errId = computeError(u0, uRef)

    varNames = ["v_x", "v_z", "b", "p"]
    fig = plt.figure(f"D{iDec}_error over time")
    for name, e, eId in zip(varNames, err.T, errId.T):
        p = plt.semilogy(e, '-', label=name, markevery=0.2)
        plt.semilogy(eId, '--', c=p[0].get_color())
    plt.legend()
    plt.grid(True)
    plt.xlabel("samples ordered with time")
    plt.ylabel("relative $L_2$ error")
    fig.set_size_inches(10, 5)
    plt.tight_layout()
    errorPlot = f"D{iDec}_error_over_time.{imgExt}"
    plt.savefig(f"{evalDir}/{errorPlot}")

    avgErr = err.mean(axis=0)
    avgErrId = errId.mean(axis=0)
    errors = pd.DataFrame(data={"model": avgErr, "id": avgErrId}, index=varNames)
    errors.loc["avg"] = errors.mean(axis=0)


    # -------------------------------------------------------------------------
    # -- Contour plots
    # -------------------------------------------------------------------------
    xGrid = dataset.infos["xGrid"][:]
    yGrid = dataset.infos["yGrid"][:]

    uI = u0[0, 2].T
    uM = uPred[0, 2].T
    uR = uRef[0, 2].T

    contourPlotSol = f"D{iDec}_contour_sol.{imgExt}"
    contourPlot(
        uM, xGrid, yGrid, title="Model output for buoyancy on first sample",
        refField=uR, refTitle="Dedalus reference",
        saveFig=f"{evalDir}/{contourPlotSol}", closeFig=True)

    contourPlotUpdate = f"D{iDec}_contour_update.{imgExt}"
    contourPlot(
        uM-uI, xGrid, yGrid, title="Model update for buoyancy on first sample",
        refField=uR-uI, refTitle="Dedalus reference",
        saveFig=f"{evalDir}/{contourPlotUpdate}", closeFig=True)

    contourPlotErr = f"D{iDec}_contour_err.{imgExt}"
    contourPlot(
        np.abs(uM-uR), xGrid, yGrid, title="Absolute error for buoyancy on first sample",
        saveFig=f"{evalDir}/{contourPlotErr}", closeFig=True)

    # -------------------------------------------------------------------------
    # -- Averaged spectrum
    # -------------------------------------------------------------------------
    sxRef, szRef = computeMeanSpectrum(uRef)
    sxPred, szPred = computeMeanSpectrum(uPred)
    k = getModes(dataset.grid[0])

    plt.figure(f"D{iDec}_spectrum")
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
    spectrumPlot = f"D{iDec}_spectrum.{imgExt}"
    plt.savefig(f"{evalDir}/{spectrumPlot}")

    plt.xlim(left=50)
    plt.ylim(top=1e-5)
    spectrumPlotHF = f"D{iDec}_spectrum_HF.{imgExt}"
    plt.savefig(f"{evalDir}/{spectrumPlotHF}")


    # -------------------------------------------------------------------------
    # -- Write slices evaluation in summary
    # -------------------------------------------------------------------------
    summary.write(TEMPLATE.format(
        iDec=iDec,
        slices=str([(sliceToStr(sX), sliceToStr(sZ)) for sX, sZ in slices]).replace("'", ""),
        errorPlot=errorPlot,
        errors=errors.to_markdown(floatfmt="1.1e"),
        contourPlotSol=contourPlotSol,
        contourPlotUpdate=contourPlotUpdate,
        contourPlotErr=contourPlotErr,
        spectrumPlot=spectrumPlot,
        spectrumPlotHF=spectrumPlotHF
        ))

summary.close()
