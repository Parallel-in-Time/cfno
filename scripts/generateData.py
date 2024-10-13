#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Dedalus simulation data
"""
import os
import sys
import numpy as np

from fnop.simulation.rbc2d import runSim, MPI_RANK
from fnop.simulation.post import OutputFiles

# Script parameters
baseDir = "generateData"
Rayleigh = 1e7
tInit = 100
resFactor = 1
nSimu = 10
dtSimu = 0.001
tEnd = 100

# Script execution
def log(msg):
    if MPI_RANK == 0:
        print(msg)

np.random.seed(12345678)
seeds = [int(s*1e6) for s in np.random.rand(nSimu)]

for seed in seeds:

    simDir = f"{baseDir}/simu_{seed}"

    # Initial run to simulate up to the quasi-stationary regime
    initRunDir = f"{simDir}/run_init"
    dtInit = 1e-2/2
    os.makedirs(initRunDir, exist_ok=True)
    log(f" -- running initial simulation with dt={dtInit:1.1e} in {initRunDir}")
    runSim(initRunDir, Rayleigh, resFactor, baseDt=dtInit, useSDC=False, tEnd=tInit,
           dtWrite=1, writeFull=True, seed=seed)
    # -- extract initial field
    initFields = OutputFiles(initRunDir).file(0)['tasks']

    # Generate simulation data
    dataRunDir = f"{simDir}/run_data"
    os.makedirs(dataRunDir, exist_ok=True)
    log(f" -- generating simulation data with dt={dtSimu:1.1e} in {dataRunDir}")
    runSim(dataRunDir, Rayleigh, 1, baseDt=dtSimu, useSDC=False,
           tEnd=tEnd, dtWrite=0.1, initFields=initFields)


# Exit if only simulation were run (using MPI for instance ...)
if len(sys.argv) > 1 and sys.argv[1] == "--runSimOnly":
    sys.exit()
