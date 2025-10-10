#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 10:47:07 2025

@author: cpf5546
"""
from cfno.data.preprocessing import HDF5Dataset
from cfno.training.pySDC import FourierNeuralOp

dataFile = "datasets/dataset_512x128_Ra1e7_dt1e-3_update.h5"
modelFile = "models/model_run24_dt1e-3.pt"


dataset = HDF5Dataset(dataFile)
model = FourierNeuralOp(checkpoint=modelFile)
