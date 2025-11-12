#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the parallel and sequential IO routines for pySDC solutions
"""
import numpy as np
from cfno.simulation.rbc2d import readPySDCSolution, writePySDCSolution, MPI_RANK, MPI_SIZE, COMM_WORLD

shape = (nVar, nX, nZ) = (4, 256, 64)

if MPI_RANK == 0:
    uRef = np.random.rand(*shape)
    np.save("uRef.npy", uRef)
COMM_WORLD.barrier()

if MPI_SIZE > 1:
    nXLoc = nX // MPI_SIZE
else:
    nXLoc = nX

uLoc = np.empty((nVar, nXLoc, nZ), dtype=np.double)

readPySDCSolution("uRef.npy", uLoc)
uLoc *= 2
writePySDCSolution("uTest.npy", uLoc, shape)

COMM_WORLD.barrier()
if MPI_RANK == 0:
    uTest = np.load("uTest.npy")
    assert np.allclose(uTest, uRef*2)
