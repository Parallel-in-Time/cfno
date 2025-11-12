#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to estimate maximum time-step size for 3D RBC simulations
"""
import numpy as np
import matplotlib.pyplot as plt

from cfno.simulation.rbc2d import runSim3D, MPI_RANK
from cfno.simulation.post import OutputFiles

from pySDC.playgrounds.dedalus.sdc import SpectralDeferredCorrectionIMEX


baseDir = "cube64_stab"
Rayleigh = 1e8

# Initial run
dirName = f"{baseDir}/run_init"
if MPI_RANK == 0:
    print(f" -- running initial simulation with baseDt in {dirName}")
runSim3D(dirName, Rayleigh, tEnd=20, dtWrite=1, writeFull=True, logEvery=10)
# -- extract initial field
initFiles = OutputFiles(dirName)
initFields = initFiles.file(0)['tasks']


# Stability for RK443
dtRK = [1e-2, 2e-2, 4e-2]
for dt in dtRK:
    dirName = f"{baseDir}/run_dt={dt:1.1g}"
    if MPI_RANK == 0:
        print(f" -- running RK443 simulation with {dt=:1.1g} in {dirName}")
    runSim3D(dirName, Rayleigh, tEnd=1, dtWrite=0.2, logEvery=10,
             baseDt=dt, initFields=initFields)

# Stability for SDC_MIN-SR-FLEX
dtSDC = [1e-2, 2e-2, 4e-2]
SpectralDeferredCorrectionIMEX.setParameters(
    nSweeps=4,
    nNodes=4,
    implSweep="MIN-SR-FLEX",
    explSweep="PIC"
    )
for dt in dtSDC:
    dirName = f"{baseDir}/run_sdc_dt={dt:1.1g}"
    if MPI_RANK == 0:
        print(f" -- running SDC simulation with {dt=:1.1g} in {dirName}")
    runSim3D(dirName, Rayleigh, tEnd=1, dtWrite=0.2, logEvery=10,
             baseDt=dt, initFields=initFields, useSDC=True)


# Plotting stuff
if MPI_RANK == 0:
    k = np.arange(32) + 0.5

    plt.figure("spectrum - RK443")
    for dt in dtRK:
        dirName = f"{baseDir}/run_dt={dt:1.1g}"
        out = OutputFiles(dirName)
        spectrum = out.getMeanSpectrum(0)
        label = f"{dt=:1.1g}"
        if np.any(np.isnan(spectrum[0][-1])):
            label += " (nan)"
        plt.loglog(k, spectrum[0][-1], label=label)
    plt.grid(True)
    plt.legend()
    plt.xlabel("wavenumber")
    plt.ylabel("u_x spectrum")

    plt.figure("spectrum - SDC")
    for dt in dtRK:
        dirName = f"{baseDir}/run_sdc_dt={dt:1.1g}"
        out = OutputFiles(dirName)
        spectrum = out.getMeanSpectrum(0)
        label = f"{dt=:1.1g}"
        if np.any(np.isnan(spectrum[0][-1])):
            label += " (nan)"
        plt.loglog(k, spectrum[0][-1], label=label)
    plt.grid(True)
    plt.legend()
    plt.xlabel("wavenumber")
    plt.ylabel("u_x spectrum")
