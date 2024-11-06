#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
sys.path.insert(2, os.getcwd())

from cfno.simulation.rbc2d import runSim, MPI_SIZE, MPI_RANK
from cfno.simulation.post import OutputFiles, extractU, contourPlot

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
    "--dtSDC", default=1e-3, type=float, help="time-step of the ref. SDC solver")
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

# Initial run
dirName = f"{runDir}/run_init"
dtInit = 1e-2/2
if MPI_RANK == 0:
    print(f" -- running initial simulation with dt={dtInit:1.1e} in {dirName}")
runSim(dirName, Rayleigh, resFactor=1, baseDt=dtInit, useSDC=False, tEnd=100,
       dtWrite=1, writeFull=True, seed=seed)
# -- extract initial field
initFiles = OutputFiles(dirName)
initFields = initFiles.file(0)['tasks']

# SDC reference solution
SpectralDeferredCorrectionIMEX.setParameters(
    implSweep="MIN-SR-FLEX", explSweep="PIC", nSweeps=nSweeps)

dirName = f"{runDir}/run_sdc_ref"
if MPI_RANK == 0:
    print(f" -- running SDC reference simulation with dt={dtSDC:1.1e} in {dirName}")
runSim(dirName, Rayleigh, resFactor=1, baseDt=dtSDC, useSDC=True,
       tEnd=tEnd, dtWrite=dtWrite, initFields=initFields)
refFiles = OutputFiles(dirName)
refFields = refFiles.file(0)['tasks']

# SDC-FNO solution
assert MPI_SIZE == 1, "cannot run FNO in space parallel (yet ...)"
SpectralDeferredCorrectionIMEX.setupNN(
    "FNOP-2", checkpoint=checkpoint, nEval=nEvalFNO)
SpectralDeferredCorrectionIMEX.setParameters(
    implSweep="MIN-SR-FLEX", explSweep="PIC", nSweeps=nSweeps)

dirName = f"{runDir}/run_sdc_fno"
print(f" -- running SDC-FNO simulation with dt={dtFNO:1.1e} in {dirName}")
runSim(dirName, Rayleigh, resFactor=1, baseDt=dtFNO, useSDC=True,
       tEnd=tEnd, dtWrite=dtWrite, initFields=initFields)
fnoFiles = OutputFiles(dirName)
fnoFields = fnoFiles.file(0)['tasks']

# Comparison
uRef = extractU(refFields, idx)
uFNO = extractU(fnoFields, idx)

diff = np.linalg.norm(uFNO-uRef, axis=(-2,-1))
norm = np.linalg.norm(uRef, axis=(-2,-1))
print(f" -- relative L2 difference for [vx, vz, b, p]:\n\t{diff/norm}")

xGrid, yGrid = refFiles.x, refFiles.y
for iVar, var in enumerate(["vx", "vz", "b", "p"]):
    contourPlot(
        uFNO[iVar].T, xGrid, yGrid, title=f"SDC-FNO for {var} after {tEnd} sec.",
        refField=uRef[iVar].T, refTitle="SDC reference",
        saveFig=f"{runDir}/comparison_{var}_{idx}.png", closeFig=True, error=False)
    print(f" -- saved {var} contour comparison for idx={idx}")
