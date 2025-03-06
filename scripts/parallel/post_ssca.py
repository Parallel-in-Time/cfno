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

cube = 128

folders = glob.glob(f"cube{cube}_N*")

data = []
for folder in folders:
    nNodes = int(folder.split('_N')[-1])
    infos = glob.glob(f"{folder}/infos_*.txt")

    for fName in infos:
        method = "sdc" if "sdc_3D" in fName else "rk3"
        with open(fName, "r") as f:
            dico = literal_eval(f.read())
        data.append({"nNodes": nNodes, "method": method, **dico})


data = pd.DataFrame(data)
data["tComp"] /= data["nSteps"]
data["tCompAll"] /= data["nSteps"]
data["tasksPerNode"] = data["MPI_SIZE"] / data["nNodes"]

os.makedirs("ppScaling", exist_ok=True)
for method in data["method"].unique():

    filePrefix = f"ppScaling/{method}"

    dataS = data[data.method == method]

    idx = dataS["tCompAll"].idxmin()
    tSeqSDC = dataS.loc[idx]["tCompAll"]

    idx = dataS.groupby(["nNodes", "MPI_SIZE"])["tComp"].idxmin()
    scaling = dataS.loc[idx]

    plt.figure(f"nNodes {method}")
    for nNodes in scaling["nNodes"].unique():
        group = scaling[scaling["nNodes"] == nNodes]
        nProcs = group["MPI_SIZE"].values
        tComp = group["tComp"].values
        plt.loglog(nProcs, tComp, 'o-', label=f"{nNodes=}")
    nProcsAll = scaling.MPI_SIZE.unique()
    plt.loglog(nProcsAll, tSeqSDC/nProcsAll, "--", c="gray")
    plt.legend()
    plt.grid(True)
    plt.xlabel("nTasks")
    plt.ylabel("tComp/step")
    plt.tight_layout()
    plt.savefig(f"{filePrefix}_nNodes.pdf")

    plt.figure(f"tasksPerNode {method}")
    for tasksPerNode in scaling["tasksPerNode"].unique():
        group = scaling[scaling["tasksPerNode"] == tasksPerNode]
        nProcs = group["MPI_SIZE"].values
        tComp = group["tComp"].values
        plt.loglog(nProcs, tComp, 'o-', label=f"tasks/node={int(tasksPerNode)}")
    nProcsAll = scaling.MPI_SIZE.unique()
    plt.loglog(nProcsAll, tSeqSDC/nProcsAll, "--", c="gray")
    plt.legend()
    plt.grid(True)
    plt.xlabel("nTasks")
    plt.ylabel("tComp/step")
    plt.tight_layout()
    plt.savefig(f"{filePrefix}_tasksPerNode.pdf")

    plt.figure(f"tasksPerNode vs nNodes {method}")
    for tasksPerNode in scaling["tasksPerNode"].unique():
        group = scaling[scaling["tasksPerNode"] == tasksPerNode]
        nProcs = group["nNodes"].values
        tComp = group["tComp"].values
        plt.loglog(nProcs, tComp, 'o-', label=f"tasks/node={int(tasksPerNode)}")
    nProcsAll = scaling.nNodes.unique()
    plt.loglog(nProcsAll, tSeqSDC/nProcsAll, "--", c="gray")
    plt.legend()
    plt.grid(True, which="major", axis="y")
    plt.grid(True, which="minor", axis="x")
    plt.xlabel("nNodes")
    plt.xticks([], major=True)
    plt.xticks(
        scaling["nNodes"].unique(), [str(n) for n in scaling["nNodes"].unique()],
        minor=True)
    plt.ylabel("tComp/step")
    plt.tight_layout()
    plt.savefig(f"{filePrefix}_tasksPerNode2.pdf")
