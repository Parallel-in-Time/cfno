#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 17:28:43 2025

@author: telu
"""
import timeit
import numpy as np
import matplotlib.pyplot as plt

from qmat.lagrange import LagrangeApproximation
from qmat.nodes import NodesGenerator

nFields = 5
dim = 3
assert dim > 0


def setup(nX):

    uBase = np.ones((nFields, dim, *[nX]*dim))

    zGrid = NodesGenerator(nodeType="CHEBY-1", quadType="GAUSS").getNodes(nX)
    zGrid += 1
    zGrid /= 2

    xGrid = np.linspace(0, 1, nX, endpoint=False)

    P = LagrangeApproximation(zGrid, weightComputation="STABLE").getInterpolationMatrix(xGrid)

    uI1 = np.einsum('ij,tvxyj->tvxyi', P, uBase)
    uI2 = (P @ uBase.reshape(-1, nX).T).T.reshape(nFields, dim, *[nX]*dim)
    assert np.allclose(uI1, uI2)
    uI3 = (P @ uBase[..., None])[..., 0]
    assert np.allclose(uI1, uI3)

    return uBase, P


def timeCall(cmd):
    tComp = timeit.timeit(cmd, globals=globals(), number=1)
    nRepeat = int(1/tComp)
    if nRepeat > 0:
        tComp += timeit.timeit(cmd, globals=globals(), number=nRepeat)
    tComp /= nRepeat+1
    return tComp


tEinsum = []
tMatmul = []
tMatmul2 = []
nXValues = [32, 64, 128, 256]
for nX in nXValues:
    uBase, P = setup(nX)
    t1 = timeCall(
        "uI1 = np.einsum('ij,tvxyj->tvxyi', P, uBase)")
    tEinsum.append(t1/(nFields*nX**dim))
    t2 = timeCall(
        "uI2 = (P @ uBase.reshape(-1, nX).T).T.reshape(nFields, dim, *[nX]*dim)")
    tMatmul.append(t2/(nFields*nX**dim))
    t3 = timeCall(
        "uI2 = (P @ uBase[..., None])[..., 0]")
    tMatmul2.append(t3/(nFields*nX**dim))

plt.loglog(nXValues, tEinsum, label=f"Einsum, nFields={nFields}")
plt.loglog(nXValues, tMatmul, label=f"Matmul, nFields={nFields}")
plt.loglog(nXValues, tMatmul2, label=f"Matmul2, nFields={nFields}")
plt.legend()
