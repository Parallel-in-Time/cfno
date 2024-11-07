#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base script to run convergence studies in space
"""
import os
import numpy as np

from cfno.simulation.rbc2d import runSim
from cfno.simulation.post import OutputFiles, checkDNS

nX, nZ = 256, 64
resFactor = 2
baseDir = f"run_M{resFactor}"

RayleighInit = 2e7
RayleighStep = 5e6
iRunOffset = 20

print(f"Running convergence study for Nx, Nz = {nX}, {nZ}")
nRunMax = 50
Rayleigh = RayleighInit
for iRun in range(nRunMax):

    iRun = iRun + iRunOffset

    dirName = f"{baseDir}/run_{iRun:02d}"
    os.makedirs(dirName, exist_ok=True)

    print(f" -- running simulation with Rayleigh={Rayleigh:1.2e} in {dirName}")
    runSim(dirName, Rayleigh, resFactor)

    out = OutputFiles(dirName)
    sMean, k = out.getFullMeanSpectrum(10)
    np.savetxt(f"{dirName}/spectrum.txt", np.vstack((sMean, k)))

    status, (a, b, c) = checkDNS(sMean, k, vRatio=4, nThrow=3)
    print(f"    {status} (a={a})")
    with open(f"{dirName}/tail.txt", "w") as f:
        f.write(f"a, b, c = {a}, {b}, {c}")

    if a > 0:
        print("Reach under-resolved space grid, existing")
        break
    else:
        Rayleigh += RayleighStep
