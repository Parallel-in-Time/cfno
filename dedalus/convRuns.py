#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base script to run convergence studies in space
"""
import numpy as np
from simu import runSimu, OutputFiles, checkDNS

nX, nZ = 256, 64
resFactor = 1
RayleighInit = 2e6
RayleighStep = 5e5

print(f"Running convergence study for Nx, Nz = {nX}, {nZ}")
nRunMax = 20
Rayleigh = RayleighInit
for iRun in range(nRunMax):

    dirName = f"run_{iRun:02d}"

    print(f" -- running simulation with Rayleigh={Rayleigh} in {dirName}")
    runSimu(dirName, Rayleigh, resFactor)

    out = OutputFiles(dirName)
    sMean, k = out.getFullMeanSpectrum(10)
    np.savetxt(f"{dirName}/spectrum.txt", np.vstack((sMean, k)))

    status, (a, b, c) = checkDNS(sMean, k)
    print(f"    {status} (a={a})")
    with open(f"{dirName}/tail.txt", "w") as f:
        f.write(f"a, b, c = {a}, {b}, {c}")

    if a > 0:
        print("Reach under-resolved space grid, existing")
        break
