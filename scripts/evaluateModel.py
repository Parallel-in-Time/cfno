#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a FNO model using some 2D RBC simulation as reference
"""
import os
import sys
sys.path.insert(2, os.getcwd())
import numpy as np

from cfno.simulation.rbc2d import runSim, MPI_RANK
from cfno.simulation.post import OutputFiles, contourPlot
from cfno.training.pySDC import FourierNeuralOp

baseDir = "evaluateModel"
Rayleigh = 1e7

MODEL_PATH = "../../model_archive/FNO2D_RBC2D_strategy2/model_nx256_nz64_dt1e_3_tin1/run2"

FNO_PARAMS = {
    "checkpoint": f"{MODEL_PATH}/model_checkpoint_80.pt"
    }

def log(msg):
    if MPI_RANK == 0:
        print(msg)

# Initial run to simulate up to the quasi-stationary regime
initRunDir = f"{baseDir}/run_init"
dtInit = 1e-2/2
os.makedirs(initRunDir, exist_ok=True)
log(f" -- running initial simulation with dt={dtInit:1.1e} in {initRunDir}")
runSim(initRunDir, Rayleigh, 1, baseDt=dtInit, useSDC=False, tEnd=100,
       dtWrite=1, writeFull=True)
# -- extract initial field
initFiles = OutputFiles(initRunDir)
initFields = initFiles.file(0)['tasks']

# Reference solution
tEnd = 5
dtRef = dtInit

refRunDir = f"{baseDir}/run_ref"
os.makedirs(refRunDir, exist_ok=True)
log(f" -- running reference simulation with dt={dtInit:1.1e} in {refRunDir}")
runSim(refRunDir, Rayleigh, 1, baseDt=dtRef, useSDC=False,
       tEnd=tEnd, dtWrite=0.1, initFields=initFields)

# Exit if only simulation were run (using MPI for instance ...)
if len(sys.argv) > 1 and sys.argv[1] == "--runSimOnly":
    sys.exit()

# Reference simulation data
refFiles = OutputFiles(refRunDir)
refFile = refFiles.file(0)
sKeys = list(refFile["scales"].keys())
gridX = refFile["scales"][sKeys[-2]][:]
gridZ = refFile["scales"][sKeys[-1]][:]
refSol = refFile['tasks']
time = refFile['scales']['sim_time'][:]
# FNO evaluation
model = FourierNeuralOp(**FNO_PARAMS)
vx0 = refFile['tasks']["velocity"][0, 0]
vz0 = refFile['tasks']["velocity"][0, 1]
b0 = refFile['tasks']["buoyancy"][0]
p0 = refFile['tasks']["pressure"][0]
u0 = np.array([vx0, vz0, b0, p0])
u1 = model.predict(u0)

# Comparison
lookingAt = "buoyancy"
if lookingAt.startswith("velocity"):
    try:
        idx = {"x": 0, "z": 1}[lookingAt[-1]]
    except KeyError:
        raise ValueError(f"wrong format for lookingAt ({lookingAt})")
    refSol = refSol["velocity"][:, idx]
    modSol = u1[idx]
elif lookingAt in ["buoyancy", "pressure"]:
    refSol = refSol[lookingAt][:]
    modSol = u1[2 if lookingAt == "buoyancy" else 3]
else:
    raise ValueError(f"wrong format for lookingAt ({lookingAt})")

contourPlot(
    modSol.T, gridX, gridZ, refField=refSol[1].T, time=time[1],
    title=f"FNO model ({lookingAt})", refTitle="Dedalus simulation",
    saveFig=f"{baseDir}/buoyancy_inference.jpg", closeFig=False)

if False:
    for i in range(refSol.shape[0]):
        contourPlot(
            refSol[i].T, gridX, gridZ, refField=refSol[i].T, time=time[1],
            title=f"FNO model ({lookingAt})", refTitle="Dedalus simulation",
            saveFig=f"{baseDir}/buoyancy_{i:02d}.jpg")
