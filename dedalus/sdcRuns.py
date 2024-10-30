#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base script to run tests with SDC
"""
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from fnop.simulation.rbc2d import runSim
from fnop.simulation.post import OutputFiles, extractU
from pySDC.playgrounds.dedalus.sdc import SpectralDeferredCorrectionIMEX


comm = MPI.COMM_WORLD
nX, nZ = 256, 64
resFactor = 1
baseDir = f"run_sdc_M{resFactor}"

Rayleigh = 1.5e7

# Initial run
dirName = f"{baseDir}/run_init"
dt = 1e-2/2
print(f" -- running initial simulation with dt={dt:1.1e} in {dirName}")
runSim(dirName, Rayleigh, resFactor, baseDt=dt, useSDC=False, tEnd=100,
       dtWrite=1, writeFull=True)
# -- extract initial field
initFiles = OutputFiles(dirName)
initFields = initFiles.file(0)['tasks']


tEnd = 1
dtRef = 1e-6
dtBase = 0.05
dtSizes = [dtBase/2**i for i in range(5)]


def error(uNum, uRef):
    refNorms = np.linalg.norm(uRef, axis=(-2, -1))
    diffNorms = np.linalg.norm(uNum-uRef, axis=(-2, -1))
    return np.mean(diffNorms/refNorms)


# Reference solution
dirName = f"{baseDir}/run_ref"
print(f" -- running reference simulation with dt={dt:1.1e} in {dirName}")
runSim(dirName, Rayleigh, resFactor, baseDt=dtRef, useSDC=False,
       tEnd=tEnd, dtWrite=tEnd/10, initFields=initFields)
refFiles = OutputFiles(dirName)
refFields = refFiles.file(0)['tasks']

uRef = extractU(refFields)


# SDC runs
SpectralDeferredCorrectionIMEX.setParameters(
    nSweeps=1,
    nNodes=4,
    implSweep="MIN-SR-FLEX",
    explSweep="PIC")


plt.figure("convergence")
errors = []
for i, dt in enumerate(dtSizes):
    dirName = f"{baseDir}/run_sdc_dt{dt:1.1e}"
    print(f" -- running SDC simulation with dt={dt:1.1e} in {dirName}")
    runSim(dirName, Rayleigh, resFactor, baseDt=dt, useSDC=True,
           tEnd=tEnd, dtWrite=tEnd, initFields=initFields)
    outFiles = OutputFiles(dirName)
    numFields = outFiles.file(0)['tasks']
    uNum = extractU(numFields)
    err = error(uNum, uRef)
    errors.append(err)
plt.loglog(dtSizes, errors, 'o-', label="SDC")


# non-SDC runs
errors = []
for i, dt in enumerate(dtSizes):
    dirName = f"{baseDir}/run_nosdc_dt{dt:1.1e}"
    print(f" -- running non-SDC simulation with dt={dt:1.1e} in {dirName}")
    runSim(dirName, Rayleigh, resFactor, baseDt=dt, useSDC=False,
           tEnd=tEnd, dtWrite=tEnd, initFields=initFields)
    outFiles = OutputFiles(dirName)
    numFields = outFiles.file(0)['tasks']
    uNum = extractU(numFields)
    err = error(uNum, uRef)
    errors.append(err)

plt.grid(True)
plt.loglog(dtSizes, errors, 'o-', label="RK443")
plt.xlabel(r"$\Delta{t}$")
plt.legend()
