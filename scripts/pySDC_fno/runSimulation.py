#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run SDC using FNO for intial guess, comparing with COPY, eventually with time parallelization
"""
import os
import matplotlib
matplotlib.use("agg")
import argparse

from cfno.data.preprocessing import HDF5Dataset
from cfno.simulation.post import contourPlot
from cfno.simulation.rbc2d import runSimPySDC, MPI_SIZE, MPI_RANK, COMM_WORLD

from pySDC.helpers.fieldsIO import Rectilinear, FieldsIO


parser = argparse.ArgumentParser(
    description='Run (parallel) SDC with and without FNO',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--dataPath", default="../datasets/dataset_pySDC_dt1e-2_update.h5",
    help="path to the dataset from which the last sample 1999 (last) is used as initial guess")
parser.add_argument(
    "--timeParallel", action="store_true", help="use time parallelization")
parser.add_argument(
    "--nSweeps", type=int, default=None, help="number of sweeps used for SDC")
parser.add_argument(
    "--nSweepsFNO", type=int, default=None, help="number of sweeps used for SDC-FNO")
args = parser.parse_args()

dataPath = args.dataPath
timeParallel = args.timeParallel
nSweeps = args.nSweeps
nSweepsFNO = args.nSweepsFNO


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
    dirName=f"{runDir}/sdc_seq", tEnd=1, restartFile=initFile, nSweeps=nSweeps,
    timeParallel=timeParallel)
if MPI_RANK == 0:
    if "tComp" in infos:
        with open(f"infos_{runDir}_SDC.txt", "w") as f:
            f.write(str(
                {"MPI_SIZE": MPI_SIZE, "timeParallel": timeParallel, "method": "SDC",
                 **infos})+"\n")



infos, _, _ = runSimPySDC(
    dirName=f"{runDir}/sdcFNO_seq", tEnd=1, restartFile=initFile,
    useFNO={"checkpoint": "model.pt"}, nSweeps=nSweepsFNO, timeParallel=timeParallel)
if MPI_RANK == 0:
    if "tComp" in infos:
        with open(f"infos_{runDir}_FNO.txt", "w") as f:
            f.write(str(
                {"MPI_SIZE": MPI_SIZE, "timeParallel": timeParallel, "method": "SDC-FNO",
                **infos})+"\n")


COMM_WORLD.Barrier()
if MPI_RANK == 0:
    Rectilinear.setupMPI(None, None, None)
    outputSDC = Rectilinear.fromFile(f"{runDir}/sdc_seq/outputs.pysdc")
    outputFNO = Rectilinear.fromFile(f"{runDir}/sdcFNO_seq/outputs.pysdc")

    contourPlot(outputFNO.readField(-1)[-1][2].T, x, y, closeFig=False,
                refField= outputSDC.readField(-1)[-1][2].T,
                title="Solution FNO", refTitle="Reference solution (SDC)",
                saveFig=f"{runDir}/solution.png")
