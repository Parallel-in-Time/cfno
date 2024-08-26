#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base script to run tests with SDC
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from rbc_simulation import runSim
from data_processing import OutputFiles
from sdc import SpectralDeferredCorrectionIMEX

nX, nZ = 256, 64
resFactor = 1
baseDir = f"run_sdc_M{resFactor}"

Rayleigh = 1.5e7

# Initial run
dirName = f"{baseDir}/run_init"
os.makedirs(dirName, exist_ok=True)
dt = 1e-2/2
if False:
    print(f" -- running initial simulation with dt={dt:1.1e} in {dirName}")
    runSim(dirName, Rayleigh, resFactor, baseDt=dt, useSDC=False, tEnd=100,
           dtWrite=1, writeFull=True)
# -- extract initial field
initFields = OutputFiles(dirName).file(0)['tasks']


# Reference solution
dirName = f"{baseDir}/run_ref"
os.makedirs(dirName, exist_ok=True)
dt = 1e-4
if False:
    print(f" -- running reference simulation with dt={dt:1.1e} in {dirName}")
    runSim(dirName, Rayleigh, resFactor, baseDt=dt, useSDC=False,
           tEnd=1, dtWrite=0.1, initFields=initFields)
refFields = OutputFiles(dirName).file(0)['tasks']


# SDC runs
SpectralDeferredCorrectionIMEX.setParameters(
    nSweep=4,
    M=4,
    implSweep=["MIN-SR-FLEX"]*4,
    explSweep="PIC")

plt.figure("convergence")

dtSizes = [1e-2/8, 1e-2/4, 1e-2/2, 1e-2]
errors = []
for i, dt in enumerate(dtSizes):
    dirName = f"{baseDir}/run_sdc_dt{dt:1.1e}"
    os.makedirs(dirName, exist_ok=True)
    if True:
        print(f" -- running SDC simulation with dt={dt:1.1e} in {dirName}")
        runSim(dirName, Rayleigh, resFactor, baseDt=dt, useSDC=True,
               tEnd=1, dtWrite=0.1, initFields=initFields)
    numFields = OutputFiles(dirName).file(0)['tasks']
    err = np.linalg.norm(
        numFields["buoyancy"][-1] - refFields["buoyancy"][-1],
        ord=np.inf)
    errors.append(err)
plt.loglog(dtSizes, errors, label="SDC")


# non-SDC runs
dtSizes = [1e-2/8, 1e-2/4, 1e-2/2, 1e-2]
errors = []
for i, dt in enumerate(dtSizes):
    dirName = f"{baseDir}/run_nosdc_dt{dt:1.1e}"
    os.makedirs(dirName, exist_ok=True)
    if True:
        print(f" -- running non-SDC simulation with dt={dt:1.1e} in {dirName}")
        runSim(dirName, Rayleigh, resFactor, baseDt=dt, useSDC=False,
               tEnd=1, dtWrite=0.1, initFields=initFields)
    numFields = OutputFiles(dirName).file(0)['tasks']
    err = np.linalg.norm(
        numFields["buoyancy"][-1] - refFields["buoyancy"][-1],
        ord=np.inf)
    errors.append(err)
plt.loglog(dtSizes, errors, label="RK443")
plt.legend()
