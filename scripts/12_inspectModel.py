#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Inspect model stored in a checkpoint',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--checkpoint", default="model.pt", help="name of the model checkpoint file")
args = parser.parse_args()

checkpoint = args.checkpoint

# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
from fnop.fno import FourierNeuralOp

model = FourierNeuralOp(checkpoint=checkpoint)
print(f"Inspecting model saved in {checkpoint} ({model.epochs} epochs) ...")

print(" -- hyper-parameters :")
for key, val in model.modelConfig.items():
    print(" "*4 + f" -- {key} : {val}")

print(" -- losses :")
print(f"     -- train : {model.losses['model']['train']}")
print(f"     -- valid : {model.losses['model']['valid']}")
