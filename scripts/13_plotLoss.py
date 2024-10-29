#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Plot loss evolution from a file storing it',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--lossesFile", default="losses.txt", help="name of the losses file")
parser.add_argument(
    "--saveFig", default="losses.pdf", help="name of the file to store the figure")
args = parser.parse_args()

lossesFile = args.lossesFile
saveFig = args.saveFig

# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
nEpochs, trainLoss, validLoss, idTrain, idValid = np.loadtxt(lossesFile).T

plt.semilogy(nEpochs, trainLoss, '-', label="train. loss")
plt.semilogy(nEpochs, validLoss, '-', label="valid. loss")
plt.semilogy(nEpochs, idTrain, "--", c="black")
plt.semilogy(nEpochs, idValid, ":", c="black")
plt.xlabel("epochs")
plt.ylabel("averaged L2 loss")
plt.legend()
plt.grid(which="major")
plt.grid(which="minor", linestyle="--")
plt.tight_layout()

if saveFig:
    plt.savefig("losses.pdf")