#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from simu import OutputFiles, checkDNS
import matplotlib.pyplot as plt


dirName = "run_M2/run_26"
sMean, k = np.loadtxt(f"{dirName}/spectrum.txt")
with open(f"{dirName}/00_infoSimu.txt", "r") as f:
    infos = f.read().strip()
infos = {val.split(" : ")[0]: val.split(" : ")[1] for val in infos.split('\n')}
infos["Rayleigh"] = float(infos["Rayleigh"])
infos["Nx, Nz"] = [int(n) for n in infos["Nx, Nz"].split(", ")]

# sMean /= np.prod(infos["Nx, Nz"])

plt.figure("spectrum")
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
plt.grid(True)
plt.xlabel("Wavenumber")
plt.ylabel("Spectrum")



Nz = infos["Nx, Nz"][1]
uxMean, uzMean, bVals = [], [], []
time = []
for i in range(out.nFiles):
    ux, uz, b = out.getMeanProfiles(i, buoyancy=True)
    time += list(out.times(i))
    uxMean.append(ux)
    uzMean.append(uz)
    bVals.append(b)
time = np.array(time)
uxMean = np.array(uxMean).reshape(-1, Nz)
uzMean = np.array(uzMean).reshape(-1, Nz)
bVals = np.array(bVals).reshape(-1, Nz)

plt.figure("mid-profile")
plt.plot(time, uxMean[:, Nz//2], label=r"$\bar{u}_x$")
plt.plot(time, uzMean[:, Nz//2], label=r"$\bar{u}_z$")
plt.legend()
plt.grid(True)
plt.xlabel("Time")
plt.ylabel("Mid-profile value")

plt.figure("profiles")
plt.plot(uxMean[500:].mean(axis=0), out.z, label=r"$\bar{u}_x$")
plt.plot(uzMean[500:].mean(axis=0), out.z, label=r"$\bar{u}_z$")
plt.plot(bVals[500:].mean(axis=0), out.z, label=r"$\bar{b}$")
plt.legend()
plt.grid(True)
plt.xlabel("Profile")
plt.ylabel("z-coordinate")
