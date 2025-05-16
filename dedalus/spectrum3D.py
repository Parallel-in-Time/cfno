#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to extract energy spectrum from 3D simulation Dedalus
"""
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt
from cfno.simulation.post import OutputFiles

parser = argparse.ArgumentParser(
    description='Compute 3D spectrum from Dedalus simulations',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "folders", nargs='+', help="folder(s) containing the simulation")
parser.add_argument(
    "--iBeg", default=0, type=int, help="starting time index")
parser.add_argument(
    "--iEnd", default=None, type=int, help="last time index (non included)")
parser.add_argument(
    "--step", default=1, type=int, help="time index step")
parser.add_argument(
    "--verbose", action="store_true", help="print computation logs")
parser.add_argument(
    "--output", default="spectrum.txt", help="file on which save the spectrum data")
parser.add_argument(
    "--yMin", default=1e-12, type=float, help="minimum value for the plot")

args = parser.parse_args()
folders = args.folders
iBeg = args.iBeg
iEnd = args.iEnd
step = args.step
verbose = args.verbose
output = args.output
yMin = args.yMin

for folder in folders:
    sFile = f"{folder}/{output}"
    if not os.path.isfile(sFile):
        if verbose: print(f" -- reading outputFiles from {folder}...")
        files = OutputFiles(folder)
        if verbose: print(" -- computing mean spectrum ...")
        data = files.getMeanSpectrum(
            0, iBeg=iBeg, iEnd=iEnd, step=step, verbose=verbose)
        spectrum = data.mean(axis=0)
        if verbose: print(f" -- saving spectrum into {sFile}...")
        np.savetxt(sFile, spectrum, header="spectrum-uv[0.1, 0.5, 0.9]")
    else:
        if verbose: print(f" -- loading spectrum from {sFile} ...")
        spectrum = np.loadtxt(sFile)

    if verbose: print(" -- ploting and saving figure to file ...")
    nK = spectrum.shape[0]
    k = np.arange(nK) + 0.5

    plt.loglog(k, spectrum.T[0], label=f"{folder} (y=0.1)")
    plt.loglog(k, spectrum.T[1], label=f"{folder} (y=0.5)")
    plt.loglog(k, spectrum.T[2], label=f"{folder} (y=0.9)")
    # kPlot = k[2:]
    # plt.loglog(kPlot, kPlot**(-6)*20, '--', c="gray")

plt.legend()
plt.xlabel("wavenumber")
plt.ylabel("energy spectrum")
plt.ylim(bottom=yMin)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.splitext(output)[0]+".pdf")
if verbose: print(" -- done !")
