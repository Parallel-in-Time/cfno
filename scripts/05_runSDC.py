#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
sys.path.insert(2, os.getcwd())

from cfno.simulation.rbc2d import runSim, MPI_SIZE, MPI_RANK
from cfno.simulation.post import OutputFiles, extractU, contourPlot, plt

from pySDC.playgrounds.dedalus.sdc import SpectralDeferredCorrectionIMEX

# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Evaluate a model on a given dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--checkpoint", default="model.pt", help="name of the file storing the model")
parser.add_argument(
    "--seed", default=1234, type=int, help="random seed for dedalus simulation")
parser.add_argument(
    "--runDir", default="sdcRun", help="directory to run the FNO-SDC simulation")
parser.add_argument(
    "--tEnd", default=1, type=float, help="simulation time interval")
parser.add_argument(
    "--dtSDC", default=1e-3, type=float, help="time-step of the base SDC solver")
parser.add_argument(
    "--dtFNO", default=1e-3, type=float, help="time-step of the SDC-FNO solver")
parser.add_argument(
    "--nEvalFNO", default=1, type=float, help="number of FNO evaluation for one prediction")
parser.add_argument(
    "--nSweeps", default=4, type=float, help="number of SDC sweeps")
parser.add_argument(
    "--dtWrite", default=1e-1, type=float, help="time-step between simulation outputs")
parser.add_argument(
    "--idx", default=-1, type=int, help="index of the output to compare FNO and SDC with")
args = parser.parse_args()

checkpoint = args.checkpoint
seed = args.seed
runDir = args.runDir
tEnd = args.tEnd
dtSDC = args.dtSDC
dtFNO = args.dtFNO
nEvalFNO = args.nEvalFNO
nSweeps = args.nSweeps
dtWrite = args.dtWrite
idx = args.idx

# -----------------------------------------------------------------------------
# SDC base settings
# -----------------------------------------------------------------------------
SpectralDeferredCorrectionIMEX.setParameters(
    nNodes=4, nodeType="LEGENDRE", quadType="RADAU-RIGHT",
    implSweep="MIN-SR-FLEX", explSweep="PIC", nSweeps=nSweeps,
    initSweep="COPY",
    )

# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
Rayleigh = 1e7


# Initial run -----------------------------------------------------------------
dirName = f"{runDir}/run_init"
dtInit = 1e-2/2
if MPI_RANK == 0:
    print(f" -- running initial simulation with dt={dtInit:1.1e} in {dirName}")
runSim(dirName, Rayleigh, resFactor=1, baseDt=dtInit, useSDC=False, tEnd=100,
       dtWrite=1, writeFull=True, seed=seed)
# -- extract initial field
initFiles = OutputFiles(dirName)
initFields = initFiles.file(0)['tasks']


# Reference solution ------------------------------------------------------
dirName = f"{runDir}/run_ref"
if MPI_RANK == 0:
    print(f" -- running SDC reference simulation with dt={dtSDC:1.1e} in {dirName}")
runSim(dirName, Rayleigh, resFactor=1, baseDt=dtSDC/100, useSDC=False,
       tEnd=tEnd, dtWrite=dtWrite, initFields=initFields)
refSolFile = OutputFiles(dirName)
refFields = refSolFile.file(0)['tasks']


# RK base solution -----------------------------------------------------------
dirName = f"{runDir}/run_rk_base"
if MPI_RANK == 0:
    print(f" -- running RK base simulation with dt={dtSDC:1.1e} in {dirName}")
runSim(dirName, Rayleigh, resFactor=1, baseDt=dtSDC, useSDC=False,
       tEnd=tEnd, dtWrite=dtWrite, initFields=initFields)
rkBaseFile = OutputFiles(dirName)
rkBaseFields = rkBaseFile.file(0)['tasks']


# SDC base solution -----------------------------------------------------------
SpectralDeferredCorrectionIMEX.setParameters(
    implSweep="MIN-SR-FLEX", explSweep="PIC", nSweeps=nSweeps)

dirName = f"{runDir}/run_sdc_base"
if MPI_RANK == 0:
    print(f" -- running SDC base simulation with dt={dtSDC:1.1e} in {dirName}")
runSim(dirName, Rayleigh, resFactor=1, baseDt=dtSDC, useSDC=True,
       tEnd=tEnd, dtWrite=dtWrite, initFields=initFields)
sdcBaseFile = OutputFiles(dirName)
sdcBaseFields = sdcBaseFile.file(0)['tasks']


# SDC-FNO solution ------------------------------------------------------------
assert MPI_SIZE == 1, "cannot run FNO in space parallel (yet ...)"
SpectralDeferredCorrectionIMEX.setupNN(
    "FNOP-2", checkpoint=checkpoint, nEval=nEvalFNO)
SpectralDeferredCorrectionIMEX.setParameters(
    implSweep="MIN-SR-FLEX", explSweep="PIC", nSweeps=nSweeps)

dirName = f"{runDir}/run_sdc_fno"
print(f" -- running SDC-FNO simulation with dt={dtFNO:1.1e} in {dirName}")
infos, _ = runSim(dirName, Rayleigh, resFactor=1, baseDt=dtFNO, useSDC=True,
       tEnd=tEnd, dtWrite=dtWrite, initFields=initFields)
sdcFNOFiles = OutputFiles(dirName)
sdcFNOFields = sdcFNOFiles.file(0)['tasks']

# FNO-only solution ------------------------------------------------------------
print(f" -- evaluating FNO only with dt={dtFNO:1.1e}")
u0 = extractU(initFields, -1)
model = SpectralDeferredCorrectionIMEX.model
nSteps = infos["nSteps"]-1  # do not count additional step to write last field
uFNO_only = []
uNext = u0
for i in range(nSteps):
    uNext = model(uNext)
    if (i+1) % int(round(dtWrite/dtFNO, ndigits=3)) == 0:
        uFNO_only.append(uNext)
uFNO_only = np.array(uFNO_only).swapaxes(0, 1)


# Error computation -----------------------------------------------------------
idx = slice(1, None)
uRef = extractU(refFields, idx)
uRK = extractU(rkBaseFields, idx)
uSDC = extractU(sdcBaseFields, idx)
uFNO = extractU(sdcFNOFields, idx)
uCopy = extractU(initFields, slice(-1, None))

def error(uRef, uNum):
    norm = np.linalg.norm(uRef, axis=(-2,-1))
    diff = np.linalg.norm(uRef-uNum, axis=(-2,-1))
    return diff/norm

errRK = error(uRef, uRK)
errSDC = error(uRef, uSDC)
errFNO = error(uRef, uFNO)
errFNO_only = error(uRef, uFNO_only)
errCopy = error(uRef, uCopy)

xValues = range(1, errRK.shape[-1]+1)
for iVar, var in enumerate(["vx", "vz", "b", "p"]):
    plt.figure(f"Error for {var}")
    for err, name in [
            (errRK, "RK443"), (errSDC, "SDC-base"), (errFNO, "SDC-FNO"),
            (errFNO_only, "FNO-only"), (errCopy, "copy"),
            ]:
        plt.semilogy(xValues, err[iVar], label=name)
    plt.legend()
    plt.xlabel("time-step")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"{runDir}/error_{var}_{idx}.pdf")

# xGrid, yGrid = sdcBaseFile.x, sdcBaseFile.y
# for iVar, var in enumerate(["vx", "vz", "b", "p"]):
#     contourPlot(
#         uFNO[iVar].T, xGrid, yGrid, title=f"SDC-FNO for {var} after {tEnd} sec.",
#         refField=uRef[iVar].T, refTitle="SDC reference",
#         saveFig=f"{runDir}/comparison_{var}_{idx}.png", closeFig=True, error=False)
#     print(f" -- saved {var} contour comparison for idx={idx}")
