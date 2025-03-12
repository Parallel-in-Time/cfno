#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-process strong scaling results
"""
import os
import glob
from ast import literal_eval
import pandas as pd
import matplotlib.pyplot as plt

folder = "juwels_scaling/cube128_32cpuPerNode"

infoFiles = glob.glob(f"{folder}/infos_*.txt")

data = []
for fName in infoFiles:
    _, _, *method, _ = os.path.basename(fName).split("_")
    method = '-'.join(method)
    with open(fName, "r") as f:
        dico = literal_eval(f.read())
    data.append({"method": method, **dico})


data = pd.DataFrame(data)
data["tComp"] /= data["nSteps"]
data["tCompAll"] /= data["nSteps"]

os.makedirs(f"{folder}/plot", exist_ok=True)

plt.figure(f"runtime {folder}")
for method in data["method"].unique():

    filePrefix = f"{folder}/plot/{method}"
    dataS = data[data.method == method]

    idx = dataS["tCompAll"].idxmin()
    tSeqSDC = dataS.loc[idx]["tCompAll"]

    idx = dataS.groupby(["MPI_SIZE"])["tComp"].idxmin()
    scaling = dataS.loc[idx]

    nProcs = scaling["MPI_SIZE"].values
    tComp = scaling["tComp"].values
    plt.loglog(nProcs, tComp, 'o-', label=method)

    nProcsAll = scaling.MPI_SIZE.unique()
    plt.loglog(nProcsAll, tSeqSDC/nProcsAll, "--", c="gray")

plt.legend()
plt.grid(True)
plt.xlabel("nTasks")
plt.ylabel("tComp/step")
plt.tight_layout()
plt.savefig(f"{filePrefix}_runtime.pdf")
