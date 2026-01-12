#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
sys.path.insert(2, os.getcwd())
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer
import torch
from cfno.utils import readConfig
from cfno.data.preprocessing import HDF5Dataset
from cfno.training.pySDC import FourierNeuralOp
from cfno.simulation.post import computeMeanSpectrum, getModes, contourPlot
from cfno.simulation.post import OutputFiles

# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Evaluate a model on a given dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--dedalusData", default="dataset.h5", help="Folder of the dataset HDF5 file from dedalus")
parser.add_argument(
    "--dataFile", default="dataset.h5", help="name of the dataset HDF5 file after preprocessing")
parser.add_argument(
    "--tSteps", default="1",type=int, help="number of autoregressive steps")
parser.add_argument(
    "--batchsize", default="1",type=int, help="number of samples")
parser.add_argument(
    "--model_dt", default="1e-3", type=float, help="model timestep")
parser.add_argument(
    "--checkpoint", default="model.pt", help="name of the file storing the model")
parser.add_argument(
    "--iSimu", default=8, type=int, help="index of the simulation to eval with")
parser.add_argument(
    "--imgExt", default="png", help="extension for figure files")
parser.add_argument(
    "--ndim", default=2, type=int, help="FNO2D or 3D")
parser.add_argument(
    "--model_class", default="CFNO", help="CFNO or FNO")
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
        args.__dict__.update(**config.train)
        if "trainDir" in config.train:
            FourierNeuralOp.TRAIN_DIR = config.train.trainDir

dataFile = args.dataFile
checkpoint = args.checkpoint
iSimu = args.iSimu
imgExt = args.imgExt
evalDir = args.evalDir
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
tSteps = args.tSteps
model_dt = args.model_dt

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
    --batchSize: {batchsize}
    --tSteps: {tSteps}

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
dedalus_dataFile = OutputFiles(args.dedalusData +'/run_data')
time_org = dedalus_dataFile.file(0)['scales']['sim_time']
model = FourierNeuralOp(checkpoint=checkpoint,model_class=args.model_class, ndim=args.ndim)
os.makedirs(evalDir, exist_ok=True)

nSamples = dataset.infos["nSamples"][()]
batchsize = args.batchsize 
nSimu = dataset.infos["nSimu"][()]
assert iSimu < nSimu, f"cannot evaluate with iSimu={iSimu} with only {nSimu} simu"
indices = slice(iSimu*nSamples, (iSimu+1)*nSamples)

# Initial solution for all samples
u0_full = dataset.inputs[indices]
# Reference solution for all samples
uRef_full = dataset.outputs[indices].copy()
if dataset.outType == "update":
    uRef_full /= dataset.outScaling
    uRef_full += u0_full

print(f'u0_full: {u0_full.shape}')
print(f'uRef_full: {uRef_full.shape}')

# Create summary file, and write header
def fmt(hdfFloat): return float(hdfFloat[()])

dtInput = fmt(dataset.infos["dtInput"])
dtSample = fmt(dataset.infos["dtSample"])
dtData = fmt(dataset.infos["dtData"])

# input solution of batchsize
# start_idx = int(np.random.randint(0,2900,1))
start_idx = 500
end_idx = int(start_idx + batchsize)
u0 = u0_full[start_idx: end_idx: 1]
print(f'u0 shape: {u0.shape}, \
       input index: {start_idx, end_idx}, \
       input time_range: {time_org[start_idx:end_idx:1]+200}')

# output solution of batchsize
out_index_start = start_idx + int((tSteps*model_dt)/dtData)
out_index_stop = end_idx + int((tSteps*model_dt)/dtData)
print(f'Ref index: ({out_index_start},{out_index_stop}), \
        output time_range: {time_org[out_index_start: out_index_stop: 1]+200}')
uRef = u0_full[out_index_start: out_index_stop: 1]



summary = open(f"{evalDir}/eval.md", "w")
summary.write(HEADER.format(
    iSimu=iSimu, checkpoint=checkpoint, dataFile=dataFile, nSamples=nSamples,
    dtInput=dtInput, dtSample=dtSample,
    outType=dataset.outType, outScaling=dataset.outScaling, batchsize=batchsize, tSteps=tSteps))

decomps = [
    [(slice(None), slice(None))],   # full domain evaluation

    # [(slice(0, 64), slice(None)),   # 4 domains distributed in X direction
    #  (slice(64, 128), slice(None)),
    #  (slice(128, 192), slice(None)),
    #  (slice(192, 256), slice(None))],

    # [(slice(None), slice(0, 32)),   # 2 domains distributed in Z direction
    #  (slice(None), slice(32, 64))],

    # [(slice(0, 64), slice(0,32)),     # 4 domains distributed in X & y direction
    #  (slice(0, 64), slice(32,64)),
    #  (slice(64, 128), slice(0,32)),
    #  (slice(64, 128), slice(32,64)),
    #  (slice(128, 192), slice(0,32)),
    #  (slice(128, 192), slice(32,64)),
    #  (slice(192, 256), slice(0,32)),
    #  (slice(192, 256), slice(32,64))],
    ]

for iDec in range(len(decomps)):
    slices = decomps[iDec]
    time = []
    uPred = np.zeros_like(uRef)
    _ = slice(None)
    print(f"Computing {tSteps}-Step prediction for D{iDec} with dt={model_dt}")
    input = u0
    for t in range(1,tSteps+1):
        for j, s in enumerate(slices):
            print(f" -- slice {j+1}/{len(slices)}")
            start_inference = default_timer()
            uPred[(_, _, *s)] = model(input[(_, _, *s)])
            stop_inference = default_timer() - start_inference
            time.append(stop_inference)
        input = uPred
    inferenceTime = np.round(sum(time),3)
    avg_inferenceTime = np.round(sum(time)/len(time),3)
    print(" -- done !")
    print(f'-- slices: {slices}')
    print(f'- -batchsize: {batchsize}')
    print(f' --shape of output: {uPred.shape}')
    print(f"-- Avg inference time on {device_name} (s) : {avg_inferenceTime}")
    print(f"-- Total inference time on {device_name} for {tSteps} iterations with dt of {model_dt} (s) : {inferenceTime}")


    # -------------------------------------------------------------------------
    # -- Relative error over time
    # -------------------------------------------------------------------------
    def norm(x):
        return np.linalg.norm(x, axis=(-2, -1))

    def computeError(uPred, uRef):
        diff = norm(uRef-uPred)
        nRef = norm(uRef)
        return diff/nRef

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
        saveFig=f"{evalDir}/{contourPlotSol}", refScales=True, closeFig=True)

    contourPlotUpdate = f"D{iDec}_contour_update.{imgExt}"
    contourPlot(
        uM-uI, xGrid, yGrid, title="Model update for buoyancy on first sample",
        refField=uR-uI, refTitle="Dedalus reference",
        saveFig=f"{evalDir}/{contourPlotUpdate}", refScales=True, closeFig=True)

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
        device=device_name,
        slices=str([(sliceToStr(sX), sliceToStr(sZ)) for sX, sZ in slices]).replace("'", ""),
        errorPlot=errorPlot,
        errors=errors.to_markdown(floatfmt="1.1e"),
        avg_inferenceTime=avg_inferenceTime,
        tSteps=tSteps,
        dt=model_dt,
        inferenceTime=inferenceTime,
        contourPlotSol=contourPlotSol,
        contourPlotUpdate=contourPlotUpdate,
        contourPlotErr=contourPlotErr,
        spectrumPlot=spectrumPlot,
        spectrumPlotHF=spectrumPlotHF
        ))

summary.close()
