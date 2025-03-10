#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run sequential SDC (no PinT) using FNO for intial guess, comparing with COPY
"""
import os
import sys
import matplotlib
matplotlib.use("agg")

from cfno.data.preprocessing import HDF5Dataset
from cfno.simulation.post import contourPlot
from cfno.simulation.rbc2d import runSimPySDC, MPI_SIZE, MPI_RANK

from pySDC.helpers.fieldsIO import Rectilinear, FieldsIO


if len(sys.argv) > 1:
    dataPath = sys.argv[1]
else:
    dataPath = "../datasets/dataset_pySDC_dt1e-2_update.h5"
if len(sys.argv) > 2:
    timeParallel = True
else:
    timeParallel = False


runDir = f"rbc2d_{MPI_SIZE:03d}"
if timeParallel:
    runDir += "_PinT"
os.makedirs(runDir, exist_ok=True)


dataset = HDF5Dataset(dataPath)
x = dataset.infos["xGrid"][()]
y = dataset.infos["yGrid"][()]
initField = dataset.inputs[1999]
contourPlot(initField[2].T, x, y,
            title="Initial Field", saveFig=f"{runDir}/initField.png")


initFile = "initFields.pysdc"
FieldsIO.ALLOW_OVERWRITE = True
file = Rectilinear(initField.dtype, initFile)
file.setHeader(4, [x, y])
file.initialize()
file.addField(0, initField)

infos, _, _ = runSimPySDC(
    dirName=f"{runDir}/sdc_seq", tEnd=1, restartFile=initFile)
if MPI_RANK == 0:
    with open(f"infos_{runDir}_SDC.txt", "w") as f:
        f.write(str(
            {"MPI_SIZE": MPI_SIZE, "timeParallel": timeParallel, "method": "SDC",
             **infos})+"\n")



infos, _, _ = runSimPySDC(dirName=f"{runDir}/sdcFNO_seq", tEnd=1, restartFile=initFile,
            useFNO={"checkpoint": "model.pt"})
if MPI_RANK == 0:
    with open(f"infos_{runDir}_FNO.txt", "w") as f:
        f.write(str(
            {"MPI_SIZE": MPI_SIZE, "timeParallel": timeParallel, "method": "SDC-FNO",
            **infos})+"\n")



if MPI_RANK == 0:
    Rectilinear.setupMPI(None, None, None)
    outputSDC = Rectilinear.fromFile(f"{runDir}/sdc_seq/outputs.pysdc")
    outputFNO = Rectilinear.fromFile(f"{runDir}/sdcFNO_seq/outputs.pysdc")

    contourPlot(outputFNO.readField(-1)[-1][2].T, x, y, closeFig=False,
                refField= outputSDC.readField(-1)[-1][2].T,
                title="Solution FNO", refTitle="Reference solution (SDC)",
                saveFig=f"{runDir}/solution.png")
