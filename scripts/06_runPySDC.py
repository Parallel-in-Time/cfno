#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import numpy as np
import matplotlib.pyplot as plt

from cfno.simulation.rbc2d import runSimPySDC
from cfno.simulation.post import OutputFiles

runDir = "pysdcRun_corr"

infos, controller, prob = runSimPySDC(f"{runDir}/run_init", tEnd=100, dtWrite=1)

files = glob.glob(f"{runDir}/run_init/*.npy")
files.sort()

times = [float(f[-12:-7]) for f in files]

means = np.array([np.abs(np.load(f)).mean(axis=1) for f in files])


dedalus = OutputFiles("sdcRun/run_init")

vxDedalus, vzDedalus, bDedalus = dedalus.getMeanProfiles(0, buoyancy=True)
zDedalus = dedalus.z


z = (prob.Z + 1)/2
z = z[0]


for name, fields in [("vx", (means[:, 0], vxDedalus)),
                     ("vz", (means[:, 1], vzDedalus)),
                     ("b", (means[:, 2], bDedalus))]:
    plt.figure(f"mean profile {name}")
    plt.plot(z, fields[0].mean(axis=0)/2, label="pySDC")
    plt.plot(zDedalus, fields[1].mean(axis=0), label="dedalus")
    plt.xlabel(name)
    plt.ylabel("z")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()