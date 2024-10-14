#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal training script
"""
import os

from fnop.trainer import Trainer

trainer = Trainer("dataset.h5", xStep=1, zStep=1)
checkPtFile = "checkpoint.pt"

trainer.switchOptimizer()
if os.path.isfile(checkPtFile):
    trainer.loadCheckpoint(checkPtFile)

# trainer.setLearningRate(0.0001)
# trainer.switchOptimizer()

trainer.runTraining(100)
trainer.saveCheckpoint(checkPtFile)
