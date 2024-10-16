#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base training script
"""
import argparse

from fnop.fno import FourierNeuralOp
from fnop.utils import readConfig

# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Train a 2D FNO model on a given dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--trainDir", default="trainDir", help="directory to store training results")
parser.add_argument(
    "--nEpochs", default=200, type=int, help="number of epochs to train on")
parser.add_argument(
    "--checkpoint", default="model.pt", help="name of the file storing the model")
parser.add_argument(
    "--saveEvery", default=100, type=int, help="save checkpoint every [...] epochs")
parser.add_argument(
    "--config", default="config.yaml", help="configuration file")
args = parser.parse_args()

config = readConfig(args.config)
if "train" in config:
    args.__dict__.update(**config.train)

sections = ["data", "model", "optim"]
for name in sections:
    assert name in config, f"config file needs a {name} section"
configs = {name: config[name] for name in sections}  # trainer class configs

nEpochs = args.nEpochs
saveEvery = args.saveEvery
checkpoint = args.checkpoint


# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
FourierNeuralOp.TRAIN_DIR = args.trainDir
model = FourierNeuralOp(**configs)
try:
    model.load(checkpoint)
except: pass

saveEvery = min(nEpochs, saveEvery)
nChunks = nEpochs // saveEvery
lastChunk = nEpochs % saveEvery

for _ in range(nChunks):
    model.learn(saveEvery)
    model.save(checkpoint)

if lastChunk > 0:
    model.learn(lastChunk)
    model.save(checkpoint)
