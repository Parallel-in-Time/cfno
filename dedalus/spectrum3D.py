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
    "folder", help="folder containing the simulation")
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
folder = args.folder
iBeg = args.iBeg
iEnd = args.iEnd
step = args.step
verbose = args.verbose
output = args.output
yMin = args.yMin

sFile = f"{folder}/{output}"
if not os.path.isfile(sFile):
    files = OutputFiles(folder)
    data = files.getMeanSpectrum(
        0, iBeg=iBeg, iEnd=iEnd, step=step, verbose=verbose)
    spectrum = np.array([d.mean(axis=0) for d in data])
    np.savetxt(sFile, spectrum, header="spectrum[uv,z]")
else:
    if verbose: print(" -- loading spectrum from file ...")
    spectrum = np.loadtxt(sFile)

if verbose: print(" -- ploting and saving figure to file ...")
nK = spectrum.shape[1]
k = np.arange(nK) + 0.5

plt.loglog(k, spectrum[0], label="uv-spectrum")
plt.loglog(k, spectrum[1], label="z-spectrum")
plt.legend()
plt.xlabel("wavenumber")
plt.ylabel("energy spectrum")
plt.ylim(bottom=yMin)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.splitext(sFile)[0]+".pdf")
if verbose: print(" -- done !")
