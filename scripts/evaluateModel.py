#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a FNO model using some 2D RBC simulation as reference
"""
import os
import sys

from fnop.simulation.rbc2d import runSim, MPI_RANK
from fnop.simulation.post import OutputFiles, contourPlot

baseDir = "evaluateModel"
Rayleigh = 1e7

def log(msg):
    if MPI_RANK == 0:
        print(msg)

# Initial run to simulate up to the quasi-stationary regime
initRunDir = f"{baseDir}/run_init"
dtInit = 1e-2/2
os.makedirs(initRunDir, exist_ok=True)
log(f" -- running initial simulation with dt={dtInit:1.1e} in {initRunDir}")
runSim(initRunDir, Rayleigh, 1, baseDt=dtInit, useSDC=False, tEnd=100,
       dtWrite=1, writeFull=True)
# -- extract initial field
initFields = OutputFiles(initRunDir).file(0)['tasks']

# Reference solution
tEnd = 1
dtRef = dtInit

refRunDir = f"{baseDir}/run_ref"
os.makedirs(refRunDir, exist_ok=True)
log(f" -- running reference simulation with dt={dtInit:1.1e} in {refRunDir}")
runSim(refRunDir, Rayleigh, 1, baseDt=dtRef, useSDC=False,
       tEnd=tEnd, dtWrite=0.1, initFields=initFields)

if len(sys.argv) > 1 and sys.argv[1] == "--runSimOnly":
    sys.exit()

# FNO evaluation and comparison with simulation data
refFile = OutputFiles(refRunDir).file(0)
sKeys = list(refFile["scales"].keys())

gridX = refFile["scales"][sKeys[-2]][:]
gridZ = refFile["scales"][sKeys[-1]][:]

bRef = refFile['tasks']["buoyancy"][:]

contourPlot(bRef[0].T, gridX, gridZ, saveFig="buoyancy.jpg")
