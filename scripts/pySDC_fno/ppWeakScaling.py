#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scaling curves for the weak scaling test with (parallel) SDC
"""
import glob
import pandas as pd
from ast import literal_eval
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

# Retrieve data
files = glob.glob("run_jusuf/infos*")
data = []
for fName in files:
    with open(fName) as f:
        dico = literal_eval(f.read())
        data.append(dico)

df = pd.DataFrame(data)

df["tComp"] /= df["nSteps"]

for tPar, tLabel, sym in zip([False, True], ["", "time-parallel "], ["o", "*"]):

    sub = df[(df["timeParallel"] == tPar)]
    sub = sub.sort_values(by="MPI_SIZE")

    nProcs = sub["MPI_SIZE"].values
    tComp = sub["tComp"].values

    plt.figure("weakScaling")
    plt.loglog(nProcs, tComp, sym+"-", label=f"{tLabel}SDC")

plt.figure("weakScaling")
plt.legend()
plt.grid(True)
plt.xlabel("$N_{cores}$")
plt.ylabel(r"$T_{wall}$ per time-step")
plt.tight_layout()
plt.ylim(0.05, 8)
plt.xlim(20, 2000)
plt.savefig("weakScaling.pdf")
