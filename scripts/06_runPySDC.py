#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from pySDC.helpers.fieldsIO import FieldsIO

from cfno.simulation.rbc2d import runSimPySDC
from cfno.training.pySDC import FourierNeuralOp

runDir = "pysdcRun_test"
dt = 1e-2
dtWrite = 0.1
dtFNO = dt

FieldsIO.ALLOW_OVERWRITE = True

def readSolutions(simDir):
    outputs = FieldsIO.fromFile(f"{runDir}/{simDir}/outputs.pysdc")
    u = np.array([outputs.readField(i)[-1] for i in range(outputs.nFields)])
    return u.swapaxes(0, 1)

def error(uRef, uNum):
    norm = np.linalg.norm(uRef, axis=(-2,-1))
    diff = np.linalg.norm(uRef-uNum, axis=(-2,-1))
    return diff/norm

# Initial run for 100 sec
infos, controller, prob = runSimPySDC(
    f"{runDir}/run_init", tEnd=1, dtWrite=1)
initFile = f"{runDir}/run_init/outputs.pysdc"

# Reference solution with very small time-step
infos, controller, prob = runSimPySDC(
    f"{runDir}/run_ref", tEnd=1, dtWrite=dtWrite, baseDt=dt,
    restartFile=initFile)

# Base solution using SDC
infos, controller, prob = runSimPySDC(
    f"{runDir}/run_sdc", tEnd=1, dtWrite=dtWrite, baseDt=dt,
    restartFile=initFile)

# Base solution using SDC (vanilla)
infos, controller, prob = runSimPySDC(
    f"{runDir}/run_sdcv", tEnd=1, dtWrite=dtWrite, baseDt=dt,
    restartFile=initFile, QI="BE", QE="FE", nSweeps=4)

# Base solution using SDC (LU)
infos, controller, prob = runSimPySDC(
    f"{runDir}/run_sdc-LU", tEnd=1, dtWrite=dtWrite, baseDt=dt,
    restartFile=initFile, QI="LU", QE="FE", nSweeps=4)

# Base solution using SDC (MIN-SR-S)
infos, controller, prob = runSimPySDC(
    f"{runDir}/run_sdc-S", tEnd=1, dtWrite=dtWrite, baseDt=dt,
    restartFile=initFile, QI="MIN-SR-S")

# Base solution using ARK3
infos, controller, prob = runSimPySDC(
    f"{runDir}/run_rk3", tEnd=1, dtWrite=dtWrite, baseDt=dt,
    restartFile=initFile, useRK=True)

# FNO only simulation
model = FourierNeuralOp(checkpoint="model.pt")
nSteps = infos["nSteps"]
uFNO_only = []
u0 = readSolutions("run_init")[:, -1]
uNext = u0[:, :, -1::-1]/2
for i in range(nSteps):
    uNext = model(uNext)
    if (i+1) % int(round(dtWrite/dtFNO, ndigits=3)) == 0:
        uFNO_only.append(uNext)
uFNO_only = np.array(uFNO_only).swapaxes(0, 1)
uFNO_only = uFNO_only[:, :, :, -1::-1]*2


uRef = readSolutions("run_ref")
uSDC = readSolutions("run_sdc")
uSDCv = readSolutions("run_sdcv")
uSDC_LU = readSolutions("run_sdc-LU")
uRK3 = readSolutions("run_rk3")
uSDC_S = readSolutions("run_sdc-S")
uCopy = u0[:, None, ...]


errSDC = error(uRef, uSDC)
errSDCv = error(uRef, uSDCv)
errRK3 = error(uRef, uRK3)
errSDC_LU = error(uRef, uSDC_LU)
errSDC_S = error(uRef, uSDC_S)

errFNO_only = error(uRef, uFNO_only)
errCopy = error(uRef, uCopy)

xValues = np.arange(1, errSDC.shape[-1]+1)*dtWrite
for iVar, var in enumerate(["vx", "vz", "b", "p"]):
    if var != "b": continue
    plt.figure(f"Error for {var} (pySDC)")
    for err, name, style in [
            (errSDC, "SDC", 's-'), (errRK3, "RK3", 'o-'),
            # (errSDCv, "SDCv", '^-'), (errSDC_LU, "SDC-LU", 'p-'), (errSDC_S, "SDC-S", '*-'),
            (errCopy, "Copy", "p-"), (errFNO_only, "FNO-only", "s-"),
            ]:
        plt.semilogy(
            xValues, err[iVar], style, label=name)
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("error")
    plt.ylim(1e-10, 1)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"{runDir}/error_{var}.pdf")



# times = [float(f[-12:-7]) for f in files]

# means = np.array([np.abs(np.load(f)).mean(axis=1) for f in files])
# means /= 2 # pySDC has a x2 scaling in space




# infos, controller, prob = runSimPySDC(
#     f"{runDir}/run_FNO2", tEnd=1, dtWrite=0.1, useFNO={"checkpoint": "model.pt"},
#     restartFile=initFile)



# dedalus = OutputFiles("sdcRun/run_init")

# vxDedalus, vzDedalus, bDedalus = dedalus.getMeanProfiles(0, buoyancy=True)
# zDedalus = dedalus.z
# tDedalus = dedalus.times()


# z = (prob.Z + 1)/2
# z = z[0]


# for name, fields in [("vx", (means[:, 0], vxDedalus)),
#                      ("vz", (means[:, 1], vzDedalus)),
#                      ("b", (means[:, 2], bDedalus)),
#                      ]:
#     plt.figure(f"mean profile {name}")
#     plt.plot(z, fields[0].mean(axis=0), label="pySDC")
#     plt.plot(zDedalus, fields[1].mean(axis=0), label="dedalus")
#     plt.xlabel("z")
#     plt.ylabel(name)
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()


# plt.figure("mid-section mean velocities")
# for name, fields in [("vx", (means[:, 0], vxDedalus)),
#                      ("vz", (means[:, 1], vzDedalus)),
#                      ]:
#     p = plt.plot(times, fields[0][:, 64//2], label=f"{name} pySDC")
#     c = p[0].get_color()
#     plt.plot(tDedalus, fields[1][:, 64//2], '--', color=c, label=f"{name} dedalus")
#     plt.xlabel("time")
#     plt.ylabel("mid-section mean vel.")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
