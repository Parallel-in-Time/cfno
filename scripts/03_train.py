#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal training script
"""
import os
import argparse

from fnop.trainer import Trainer
from fnop.utils import read_config

# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Train a 2D FNO model on a given dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--nEpochs", default=10, type=int, help="number of epochs to train on")
parser.add_argument(
    "--checkpoint", default="model.pt", help="name of the file storing the model")
parser.add_argument(
    "--config", default="config.yaml", help="configuration file")
args = parser.parse_args()

config = read_config(args.config)

sections = ["data", "model", "optim"]
for name in sections:
    assert name in config, f"config file needs a {name} section"
params = {name: config[name] for name in sections}  # trainer class parameters

if "train" in config:
    args.__dict__.update(**config.train)

nEpochs = args.nEpochs
checkpoint = args.checkpoint

# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
trainer = Trainer(**params)

if os.path.isfile(checkpoint):
    trainer.load(checkpoint)
trainer.learn(nEpochs)
trainer.save(checkpoint)
