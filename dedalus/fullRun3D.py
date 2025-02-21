#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for a full run (with output) for 3D RBC simulations
"""
from cfno.simulation.rbc2d import runSim3D

Ra = 1e8
resFactor = 2

runSim3D(
    f"cube{64*resFactor}", Rayleigh=Ra, resFactor=resFactor,
    baseDt=2e-2, logEvery=10,
    )
