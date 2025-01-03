#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from cfno.simulation.rbc2d import runSim, MPI_RANK, MPI_SIZE, SpectralDeferredCorrectionIMEX

parser = argparse.ArgumentParser(
    description='Run scaling test with dedalus',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--useSDC", action="store_true", help="use SDC time integrator")
parser.add_argument(
    "--timeParallel", action="store_true", help="use SDC time-parallel integrator")
args = parser.parse_args()

tEnd = 1
useSDC = args.useSDC or args.timeParallel
timeParallel = args.timeParallel

dirID = f"{MPI_SIZE:03d}"
if useSDC: dirID += "_sdc"
if timeParallel: dirID += "_tPar"

SpectralDeferredCorrectionIMEX.setParameters(
    nNodes=4, implSweep="MIN-SR-FLEX", explSweep="PIC", initSweep="COPY"
)

infos, solver, b = runSim(
    f"scaling_{dirID}",
    tEnd=tEnd, dtWrite=2*tEnd, writeSpaceDistr=True, logEvery=10000,
    useSDC=useSDC, timeParallel=timeParallel)
if MPI_RANK == 0:
    with open(f"infos_{dirID}.txt", "w") as f:
        f.write(str(infos)+"\n")
