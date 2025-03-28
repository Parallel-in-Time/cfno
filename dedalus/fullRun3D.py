#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for a full run (with output) for 3D RBC simulations
"""
import sys
from cfno.simulation.rbc3d import runSim3D

simDir = "."
if len(sys.argv) > 1:
    simDir = sys.argv[1]

Ra = 1e8
resFactor = 1

runSim3D(
    f"{simDir}/cube{64*resFactor}", Rayleigh=Ra, resFactor=resFactor,
    baseDt=1e-2/2, logEvery=10, tEnd=120, dtWrite=0.5, writeFull=True,
    )
