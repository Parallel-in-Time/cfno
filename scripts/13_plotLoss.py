#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
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
    "--yMin", default=None, type=float, help="minimum value in y axis")
parser.add_argument(
    "--yMax", default=None, type=float, help="maximum value in y axis")
parser.add_argument(
    "--xMin", default=None, type=float, help="minimum value in x axis")
parser.add_argument(
    "--xMax", default=None, type=float, help="maximum value in x axis")
parser.add_argument(
    "--scaleLoss", action="store_true", help="scale the loss by loss[0]"
)
parser.add_argument(
    "--saveFig", default="losses.pdf", help="name of the file to store the figure")
parser.add_argument(
    "--trainDirs", nargs='*', default=["."], help="training directories to plot the loss from")
args = parser.parse_args()

lossesFile = args.lossesFile
saveFig = args.saveFig
yMin, yMax = args.yMin, args.yMax
xMin, xMax = args.xMin, args.xMax
scaleLoss = args.scaleLoss

# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
currentDir = os.getcwd()

for trainDir in args.trainDirs:

    os.chdir(trainDir)

    data = np.loadtxt(lossesFile).T
    nEpochs, trainLoss, validLoss, idTrain, idValid = data[:5]
    if scaleLoss:
        idTrain /= trainLoss[0]
        trainLoss /= trainLoss[0]
        idValid /= validLoss[0]
        validLoss /= validLoss[0]

    plt.figure()
    plt.semilogy(nEpochs, trainLoss, '-', label="train. loss")
    plt.semilogy(nEpochs, validLoss, '-', label="valid. loss")
    plt.semilogy(nEpochs, idTrain, "--", c="black")
    plt.semilogy(nEpochs, idValid, ":", c="black")
    plt.xlabel("epochs")
    plt.ylabel("L2 loss")
    plt.legend()
    plt.ylim(yMin, yMax)
    plt.xlim(xMin, xMax)
    plt.grid(which="major")
    plt.grid(which="minor", linestyle="--")
    plt.tight_layout()

    if saveFig:
        plt.savefig("losses.pdf")

    if len(data) > 5:
        # gradient and tComp are also in losses file
        gradient, tComp = data[5:]
        plt.figure()
        plt.semilogy(nEpochs, gradient, '-', label="gradient norm")
        plt.xlabel("epochs")
        plt.ylabel("norm")
        plt.legend()
        plt.grid(which="major")
        plt.grid(which="minor", linestyle="--")
        plt.tight_layout()
        if saveFig:
            plt.savefig("gradient.pdf")

        plt.figure()
        plt.plot(nEpochs, tComp, '-', label="tComp per epoch")
        plt.xlabel("epochs")
        plt.ylabel("seconds")
        plt.legend()
        plt.grid(which="major")
        plt.grid(which="minor", linestyle="--")
        plt.tight_layout()
        if saveFig:
            plt.savefig("tComp.pdf")

    os.chdir(currentDir)

