#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small script to investigate the step update between two solution fields
"""
from fnop.data import HDF5Dataset
from fnop.simulation.post import contourPlot

dataset = HDF5Dataset("dataset.h5")

xGrid = dataset.file['infos/xGrid'][:]
zGrid = dataset.file['infos/zGrid'][:]

iVar = 2
iSample = 999
factor = 20


uInpt, uOutp = dataset[iSample]
delta = uOutp-uInpt
delta *= factor

contourPlot(
    uInpt[iVar].T, xGrid, zGrid, title="Field",
    refTitle="Delta", refField=delta[iVar].T, closeFig=False)
