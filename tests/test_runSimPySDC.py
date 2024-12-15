#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test pySDC run routine
"""
import numpy as np
from cfno.simulation.rbc2d import runSimPySDC, MPI_RANK, MPI_SIZE

initFile = "runSeq/sol_000.0sec.npy"

tEnd = 0.2

if MPI_SIZE == 1:
    # One-shot sequential run
    runSimPySDC("runSeq", tBeg=0, tEnd=tEnd, dtWrite=tEnd)

    # One-shot sequential run, from restart
    runSimPySDC("runSeqRes", tBeg=0, tEnd=tEnd, dtWrite=tEnd, restartFile=initFile)

    # Two-shot sequential run, from restart
    runSimPySDC("runSeqRes1", tBeg=0, tEnd=tEnd/2, dtWrite=tEnd/2, restartFile=initFile)
    runSimPySDC("runSeqRes2", tBeg=tEnd/2, tEnd=tEnd, dtWrite=tEnd/2, restartFile="runSeqRes1/sol_000.1sec.npy")

else:
    # Parallel run, from restart
    runSimPySDC("runPar", tBeg=0, tEnd=tEnd, dtWrite=tEnd, restartFile=initFile)

if MPI_RANK == 0:
    try:
        uSeq0 = np.load("runSeq/sol_000.0sec.npy")
        uSeq1 = np.load("runSeq/sol_000.2sec.npy")

        uSeqRes0 = np.load("runSeqRes/sol_000.0sec.npy")
        uSeqRes1 = np.load("runSeqRes/sol_000.2sec.npy")

        uSeqRes2 = np.load("runSeqRes2/sol_000.2sec.npy")
    except:
        raise ValueError("requires test run in sequential first")
    
    assert np.allclose(uSeq0, uSeqRes0)
    assert np.allclose(uSeq1, uSeqRes1)
    assert np.allclose(uSeq1, uSeqRes2)
    
    if MPI_SIZE > 1:

        uPar0 = np.load("runPar/sol_000.0sec.npy")
        uPar1 = np.load("runPar/sol_000.2sec.npy")
        
        assert np.allclose(uSeq0, uPar0)
        assert np.allclose(uSeq1, uPar1)
