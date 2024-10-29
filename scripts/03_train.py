#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base training script
"""
import os
import sys
sys.path.insert(2, os.getcwd())
import argparse

from fnop.training.fno_pysdc import FourierNeuralOp
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
    "--savePermanent", action="store_true", help="save permanent checkpoint into [...]_epochs[...].pt files")
parser.add_argument(
    "--noTensorboard", action="store_false", help="do not use tensorboard for losses output (only native)")
parser.add_argument(
    "--logFile", default=FourierNeuralOp.LOG_FILE, help='log file name (use "" for no log in file)')
parser.add_argument(
    "--noLogStdout", action="store_false", help="do not log training file in stdout")
parser.add_argument(
    "--config", default="config.yaml", help="configuration file")
args = parser.parse_args()

config = readConfig(args.config)
if "train" in config:
    args.__dict__.update(**config.train)

sections = ["data", "model", "optim", "lr_scheduler"]
for name in sections:
    assert name in config, f"config file needs a {name} section"
configs = {name: config[name] for name in sections}  # trainer class configs

nEpochs = args.nEpochs
saveEvery = args.saveEvery
savePermanent = args.savePermanent
checkpoint = args.checkpoint


# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
FourierNeuralOp.TRAIN_DIR = args.trainDir
FourierNeuralOp.USE_TENSORBOARD = args.noTensorboard
FourierNeuralOp.LOG_FILE = args.logFile
FourierNeuralOp.LOG_STDOUT = args.noLogStdout

model = FourierNeuralOp(**configs)
try:
    model.load(checkpoint)
except: pass

saveEvery = min(nEpochs, saveEvery)
nChunks = nEpochs // saveEvery
lastChunk = nEpochs % saveEvery

cPrefix = os.path.splitext(checkpoint)[0]

for _ in range(nChunks):
    model.learn(saveEvery)
    model.save()

if lastChunk > 0:
    model.learn(lastChunk)
    model.save()