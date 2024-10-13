#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Dedalus simulation data
"""
import os
import sys
import argparse
import numpy as np

from fnop.simulation.rbc2d import runSim, MPI_RANK
from fnop.simulation.post import OutputFiles

# Script parameters
parser = argparse.ArgumentParser(
    description='Create training (and validation) dataset from Dedalus simulation data',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--dataDir", default="generateData", help="dir. containing simulation data")
parser.add_argument(
    "--Rayleigh", default=1e7, type=float, help="Rayleigh number")
parser.add_argument(
    "--tInit", default=100, type=float, help="initial simulation time (before generating data)")
parser.add_argument(
    "--resFactor", default=1, type=int, help="resolution factor for space grid (256*resFactor, 64*resFactor)")
parser.add_argument(
    "--nSimu", default=10, type=int, help="number of simulation with different seeds")
parser.add_argument(
    "--dtSimu", default=0.001, type=float, help="base time-step for simulation")
parser.add_argument(
    "--dtData", default=0.1, type=float, help="time-step for simulation data output")
parser.add_argument(
    "--tEnd", default=100, type=float, help="simulation time for data generation")
parser.add_argument(
    "--seed", default=12345678, type=int, help="starting seed to generate all simulation seeds")
args = parser.parse_args()

dataDir = args.dataDir
Rayleigh = args.Rayleigh
tInit = args.tInit
resFactor = args.resFactor
nSimu = args.nSimu
dtSimu = args.dtSimu
dtData = args.dtData
tEnd = args.tEnd
np.random.seed(args.seed)

# Script execution
def log(msg):
    if MPI_RANK == 0:
        print(msg)

seeds = [int(s*1e6) for s in np.random.rand(nSimu)]

for seed in seeds:

    simDir = f"{dataDir}/simu_{seed}"

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
    log(f" -- generating simulation data with dt={dtData:1.1e} in {dataRunDir}")
    runSim(dataRunDir, Rayleigh, 1, baseDt=dtSimu, useSDC=False,
           tEnd=tEnd, dtWrite=dtData, initFields=initFields)
