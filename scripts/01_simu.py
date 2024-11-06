#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
sys.path.insert(2, os.getcwd())
from cfno.simulation.rbc2d import runSim, MPI_RANK
from cfno.simulation.post import OutputFiles
from cfno.utils import readConfig

# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Generate Dedalus simulation data for 2D RBC',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--dataDir", default="simuData", help="directory containing simulation data")
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
parser.add_argument(
    "--config", default=None, help="config file, overwriting all parameters specified in it")
args = parser.parse_args()

if args.config is not None:
    config = readConfig(args.config)
    assert "simu" in config, "config file needs a simu section"
    args.__dict__.update(**config.simu)

dataDir = args.dataDir
Rayleigh = args.Rayleigh
tInit = args.tInit
resFactor = args.resFactor
nSimu = args.nSimu
dtSimu = args.dtSimu
dtData = args.dtData
tEnd = args.tEnd
np.random.seed(args.seed)

# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
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
           dtWrite=1, writeFull=True, tBeg=0,seed=seed)
    # -- extract initial field
    initFiles = OutputFiles(initRunDir)
    initFields = initFiles.file(0)['tasks']

    # Generate simulation data
    dataRunDir = f"{simDir}/run_data"
    os.makedirs(dataRunDir, exist_ok=True)
    log(f" -- generating simulation data with dt={dtData:1.1e} (dtSimu={dtSimu:1.1e}) in {dataRunDir}")
    runSim(dataRunDir, Rayleigh, 1, baseDt=dtSimu, useSDC=False,
           tEnd=tEnd, dtWrite=dtData, initFields=initFields)
