#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal training script
"""
import os

from fnop.trainer import Trainer


config = {
    "trainConfig": {
        "dataFile": "dataset.h5",
        "trainRatio": 0.8,
        "seed": None,
        "batchSize": 20,
        "xStep": 1,
        "yStep": 1,
        },
    "modelConfig": {
        "da": 4, "du": 4, "dv": 4, "kX": 8, "kY": 8,
        "nLayers": 2, "forceFFT": False
        },
    "optimConfig": {
        "name": "adam",
        "lr": 0.0001,
        "weight_decay": 1e-5
        }
    }
checkPtFile = "checkpoint.pt"

trainer = Trainer(**config)
if os.path.isfile(checkPtFile):
    trainer.loadCheckpoint(checkPtFile)

trainer.runTraining(10)
trainer.saveCheckpoint(checkPtFile)
