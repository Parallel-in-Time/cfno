#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from cfno.simulation.rbc2d import runSim, \
    MPI_RANK, MPI_SIZE, SpectralDeferredCorrectionIMEX
from cfno.simulation.rbc3d import runSim3D


parser = argparse.ArgumentParser(
    description='Run scaling test with dedalus',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--useSDC", action="store_true", help="use SDC time integrator")
parser.add_argument(
    "--timeParallel", action="store_true", help="use SDC time-parallel integrator")
parser.add_argument(
    "--groupTime", action="store_true", help="group time-parallel processes")
parser.add_argument(
    "--useTimePar2", action="store_true", help="use time-parallel implementation #2")
parser.add_argument(
    "--tEnd", type=float, default=1.0, help="final time of simulation")
parser.add_argument(
    "--dt", type=float, default=0.005, help="base time-step for the simulation")
parser.add_argument(
    "--run3D", action="store_true", help="use 3D simulation")
parser.add_argument(
    "--resFactor", type=int, default=1, help="factor for base grid sizes (256x64 for 2D, 64^3 for 3D)")
parser.add_argument(
    "--mpiBlocks", type=int, default=None, nargs='*', help="number of MPI blocks in Y and Z for 3D simulation")

args = parser.parse_args()

useSDC = args.useSDC or args.timeParallel
timeParallel = args.timeParallel
groupTime = args.groupTime
useTimePar2 = args.useTimePar2
tEnd = args.tEnd
dt = args.dt
run3D = args.run3D
resFactor = args.resFactor
mpiBlocks = args.mpiBlocks

dirID = f"{MPI_SIZE:04d}"
if mpiBlocks is not None:
    dirID += f"-[{','.join(str(n) for n in mpiBlocks)}]"
if useSDC:
    dirID += "_sdc"
else:
    dirID += "_rk3"
if timeParallel:
    dirID += "_tPar"
    if useTimePar2:
        dirID += "2"
    if groupTime:
        dirID += "G"
if run3D:
    dirID += "_3D"

SpectralDeferredCorrectionIMEX.setParameters(
    nNodes=4, implSweep="MIN-SR-FLEX", explSweep="PIC", initSweep="COPY"
)
runFunction = runSim3D if run3D else runSim
arguments = dict(
    dirName=f"scaling_{dirID}", resFactor=resFactor, tEnd=tEnd, baseDt=dt,
    dtWrite=2*tEnd, writeSpaceDistr=True, logEvery=10000,
    useSDC=useSDC, timeParallel=timeParallel, groupTimeProcs=groupTime,
    useTimePar2=useTimePar2)
if run3D:
    arguments["mpiBlocks"] = mpiBlocks
    arguments["writeFields"] = False
infos, solver = runFunction(**arguments)
if MPI_RANK == 0:
    with open(f"infos_{dirID}.txt", "w") as f:
        f.write(str(infos)+"\n")
