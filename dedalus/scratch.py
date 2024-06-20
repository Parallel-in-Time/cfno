#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from simu import OutputFiles, checkDNS
import matplotlib.pyplot as plt


dirName = "run_M2/run_20"
sMean, k = np.loadtxt(f"{dirName}/spectrum.txt")
with open(f"{dirName}/00_infoSimu.txt", "r") as f:
    infos = f.read().strip()
infos = {val.split(" : ")[0]: val.split(" : ")[1] for val in infos.split('\n')}
infos["Rayleigh"] = float(infos["Rayleigh"])
infos["Nx, Nz"] = [int(n) for n in infos["Nx, Nz"].split(", ")]

# sMean /= np.prod(infos["Nx, Nz"])

sMean /= infos["Nx, Nz"][0]
plt.loglog(k[:-1], sMean[:-1], label=f"Ra={infos['Rayleigh']:1.1e}")


out = OutputFiles(dirName)

vRatio = 4
nThrow = 3

status, (a, b, c) = checkDNS(sMean, k, vRatio, nThrow)
print(f"{status} (a={a})")

nValues = k.size//vRatio
y = np.log(sMean[-nValues-nThrow:-nThrow])
x = np.log(k[-nValues-nThrow:-nThrow])

y = a*x**2 + b*x + c
x = np.exp(x)
y = np.exp(y)
plt.plot(x, y, 'k--')

plt.legend()
