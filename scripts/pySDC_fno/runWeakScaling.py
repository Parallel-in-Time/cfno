#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run (parallel) SDC with a given resolution factor
"""
import os
import matplotlib
matplotlib.use("agg")
import argparse

from cfno.simulation.rbc2d import runSimPySDC, MPI_SIZE, MPI_RANK
from pySDC.helpers.fieldsIO import FieldsIO

FieldsIO.ALLOW_OVERWRITE = True

parser = argparse.ArgumentParser(
    description='Run (parallel) SDC with a given resolution factor',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--timeParallel", action="store_true", help="use time parallelization")
parser.add_argument(
    "--resFactor", default=1, type=int, help="resolution factor in space")
args = parser.parse_args()

timeParallel = args.timeParallel
resFactor = args.resFactor


runDir = f"rbc2d_{MPI_SIZE:03d}_R{resFactor:02d}_SDC"
if timeParallel:
    runDir += "_PinT"
os.makedirs(runDir, exist_ok=True)


infos, _, _ = runSimPySDC(
    dirName=f"{runDir}/sdc_seq", tEnd=0.2, nSweeps=4, dtWrite=0.2,
    resFactor=resFactor, timeParallel=timeParallel)
if MPI_RANK == 0:
    if "tComp" in infos:
        with open(f"infos_{runDir}.txt", "w") as f:
            f.write(str(
                {"MPI_SIZE": MPI_SIZE, "timeParallel": timeParallel, **infos})+"\n")
