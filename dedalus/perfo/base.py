#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base script to plot the
"""
import numpy as np
import matplotlib.pyplot as plt

# General matplotlib settings
plt.rc('font', size=12)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['xtick.major.pad'] = 5
plt.rcParams['ytick.major.pad'] = 5
plt.rcParams['axes.labelpad'] = 6
plt.rcParams['markers.fillstyle'] = 'none'
plt.rcParams['lines.markersize'] = 7.0
plt.rcParams['lines.markeredgewidth'] = 1.5
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['figure.max_open_warning'] = 100

# Data (parallel SDC, MIN-SR-FLEX, K=4, M=4, LEGENDRE/RADAU-RIGHT, COPY, LASTNODE)
nX, nZ, nSteps = 512, 128, 1/(1e-2/4)
tComp = np.array([456.6, 269.1, 132.3, 70.97, 42.08, 29.43, 20.64, 18])
nProc = np.array([1, 2, 4, 8, 16, 32, 64, 128])

tScaled = tComp/(nX*nZ*nSteps)
speedup = tComp[0]/nProc[0]/tComp
efficiency = speedup/nProc

plt.figure("tScaled")
plt.loglog(nProc, tScaled, 'o-')
plt.loglog(nProc, tScaled[0]/nProc[0]/nProc, '--', c='gray')
plt.text(1.1, 7e-7, f"Max. speedup : {speedup.max():1.2f}")
plt.ylabel("Scaled runtime $t_{wall}/N_{dof}/N_{steps}$")

plt.figure("speedup")
plt.loglog(nProc, speedup, 'o-')
plt.loglog(nProc, nProc, '--', c='gray')
plt.ylabel("Parallel Speedup")

plt.figure("efficiency")
plt.semilogx(nProc, efficiency, 'o-')
plt.semilogx(nProc, nProc*0+1, '--', c='gray')
plt.ylim(0, 1.1)
plt.ylabel("Parallel Efficiency")


for figName in ["tScaled", "speedup", "efficiency"]:
    plt.figure(figName)
    plt.xlabel("$N_{proc}$")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{figName}.pdf")
