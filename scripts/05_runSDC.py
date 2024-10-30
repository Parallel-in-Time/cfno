#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
sys.path.insert(2, os.getcwd())

from fnop.simulation.rbc2d import runSim, MPI_SIZE, MPI_RANK
from fnop.simulation.post import OutputFiles, extractU

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
args = parser.parse_args()

checkpoint = args.checkpoint
seed = args.seed
runDir = args.runDir

# -----------------------------------------------------------------------------
# SDC base settings
# -----------------------------------------------------------------------------
SpectralDeferredCorrectionIMEX.setParameters(
    nNodes=4, nodeType="LEGENDRE", quadType="RADAU-RIGHT",
    implSweep="MIN-SR-FLEX", explSweep="PIC", nSweeps=4,
    initSweep="COPY",
    )

# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
Rayleigh = 1e7

# Initial run
dirName = f"{runDir}/run_init"
dt = 1e-2/2
if MPI_RANK == 0:
    print(f" -- running initial simulation with dt={dt:1.1e} in {dirName}")
runSim(dirName, Rayleigh, resFactor=1, baseDt=dt, useSDC=False, tEnd=100,
       dtWrite=1, writeFull=True, seed=seed)
# -- extract initial field
initFiles = OutputFiles(dirName)
initFields = initFiles.file(0)['tasks']

# SDC and SDC-FNO runs ...
tEnd = 1
dt = 1e-3

# SDC reference solution
dirName = f"{runDir}/run_sdc_ref"
if MPI_RANK == 0:
    print(f" -- running SDC reference simulation with dt={dt:1.1e} in {dirName}")
runSim(dirName, Rayleigh, resFactor=1, baseDt=dt, useSDC=True,
       tEnd=tEnd, dtWrite=tEnd/100, initFields=initFields)
refFiles = OutputFiles(dirName)
refFields = refFiles.file(0)['tasks']

# SDC-FNO solution
assert MPI_SIZE == 1, "cannot run FNO in space parallel (yet ...)"
SpectralDeferredCorrectionIMEX.setupNN("FNOP-2", checkpoint=checkpoint)

dirName = f"{runDir}/run_sdc_fno"
print(f" -- running SDC-FNO simulation with dt={dt:1.1e} in {dirName}")
runSim(dirName, Rayleigh, resFactor=1, baseDt=dt, useSDC=True,
       tEnd=tEnd, dtWrite=tEnd/100, initFields=initFields)
fnoFiles = OutputFiles(dirName)
fnoFields = refFiles.file(0)['tasks']

# Comparison
uRef = extractU(refFields)
uFNO = extractU(fnoFields)
