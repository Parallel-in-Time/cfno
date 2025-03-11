#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scaling curves for the strong and weak scaling test of (parallel) SDC (FNO)
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
files = glob.glob("run_jureca/infos*")
data = []
for fName in files:
    with open(fName) as f:
        dico = literal_eval(f.read())
        data.append(dico)

df = pd.DataFrame(data)
df.fillna(0, inplace=True)

df["tComp"] /= df["nSteps"]
df["tModelEval"] /= df["nSteps"]

df["tCompAll"] = df["tComp"]*df["MPI_SIZE"]

idx = df["tCompAll"].idxmin()
tSeqMin = df.loc[idx]["tCompAll"]

for tPar, tLabel, ls in zip([False, True], ["", "time-parallel "], ["-", "--"]):
    for method, sym in zip(df["method"].unique(), ["o", "*"]):

        sub = df[(df["timeParallel"] == tPar) & (df["method"] == method)]
        sub = sub.sort_values(by="MPI_SIZE")

        nProcs = sub["MPI_SIZE"].values
        tComp = sub["tComp"].values

        plt.figure("timings")
        plt.loglog(nProcs, tComp, sym+ls, label=f"{tLabel}{method}")

        plt.figure("speedup")
        plt.loglog(nProcs, tSeqMin/tComp, sym+ls, label=f"{tLabel}{method}")

        plt.figure("tModelEval")
        if method == "SDC-FNO":
            tModelEval = sub["tModelEval"].values
            plt.loglog(nProcs, tModelEval, sym+ls, label=f"{tLabel}{method}")

        plt.figure("tSpeedup")
        if tPar:
            seq = df[(df["timeParallel"] == False) & (df["method"] == method)]
            seq = seq.sort_values(by="sSize")
            pint = sub.sort_values(by="sSize")
            sSize = pint["sSize"].values
            tSpeedup = (seq["tComp"].values[:pint.shape[0]] / sub["tComp"].values)
            plt.semilogx(sSize, tSpeedup, sym+ls, label=f"{tLabel}{method}")




plt.figure("timings")
nProcsAll = sorted(df.MPI_SIZE.unique())
plt.loglog(nProcsAll, tSeqMin/nProcsAll, "--", c="gray")
plt.legend()
plt.grid(True)
plt.xlabel("$N_{cores}$")
plt.ylabel(r"$T_{wall}$ per time-step")
plt.tight_layout()

plt.figure("speedup")
plt.loglog(nProcsAll, nProcsAll, "--", c="gray")
plt.legend()
plt.grid(True)
plt.xlabel("$N_{cores}$")
plt.ylabel("Speedup")
plt.tight_layout()

plt.figure("tModelEval")
plt.legend()
plt.grid(True)
plt.xlabel("$N_{cores}$")
plt.ylabel("GPU time per step")
plt.tight_layout()

plt.figure("tSpeedup")
plt.legend()
plt.grid(True)
plt.xlabel("$N_{cores}$ in space")
plt.ylabel("Time-parallel speedup")
plt.tight_layout()
