#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a FNO model using some 2D RBC simulation as reference
"""
import os
import sys

import numpy as np

from fnop.simulation.rbc2d import runSim, MPI_RANK
from fnop.simulation.post import OutputFiles, contourPlot
from fnop.inference.inference import FNOInference

baseDir = "evaluateModel"
Rayleigh = 1e7

MODEL_PATH = "../../model_archive/FNO2D_RBC2D_strategy2/model_nx256_nx64_dt1e_1_tin1"

FNO_PARAMS = {
    "config": f"{MODEL_PATH}/fno2d_strat2_dt1e-1.yaml",
    "checkpoint": f"{MODEL_PATH}/model_checkpoint_1499.pt"
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
initFields = OutputFiles(initRunDir).file(0)['tasks']

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
refFile = OutputFiles(refRunDir).file(0)
sKeys = list(refFile["scales"].keys())
gridX = refFile["scales"][sKeys[-2]][:]
gridZ = refFile["scales"][sKeys[-1]][:]
refSol = refFile['tasks']

# FNO evaluation
model = FNOInference(**FNO_PARAMS)
vx0 = refFile['tasks']["velocity"][0, 0]
vz0 = refFile['tasks']["velocity"][0, 1]
b0 = refFile['tasks']["buoyancy"][0]
p0 = refFile['tasks']["pressure"][0]
u0 = np.array([vx0, vz0, b0, p0])
u1 = model.predict(u0)

# Comparison
lookingAt = "velocity-z"
if lookingAt.startswith("velocity"):
    try:
        idx = {
            "x": 0,
            "z": 1,
        }[lookingAt[-1]]
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
    modSol.T, gridX, gridZ, refField=refSol[1].T,
    title=f"FNO model ({lookingAt})", refTitle="Dedalus simulation",
    saveFig=f"{baseDir}/buoyancy_inference.jpg", closeFig=False)

if False:
    for i in range(refSol.shape[0]):
        contourPlot(
            refSol[i].T, gridX, gridZ, refField=refSol[i].T,
            title=f"FNO model ({lookingAt})", refTitle="Dedalus simulation",
            saveFig=f"{baseDir}/buoyancy_{i:02d}.jpg")
