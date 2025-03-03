#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to extract profiles from 3D simulation Dedalus
"""
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt
from cfno.simulation.post import OutputFiles

parser = argparse.ArgumentParser(
    description='Compute 3D profiles from Dedalus simulations',
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
    "--output", default="profiles.txt", help="file on which save the spectrum data")

args = parser.parse_args()
folders = args.folders
iBeg = args.iBeg
iEnd = args.iEnd
step = args.step
verbose = args.verbose
output = args.output

fig, ax = plt.subplots(layout='constrained')
ax.set_ylabel("Temp. fluctuations")
ax.set_ylabel("$z$")
shift = 0.4
ax2 = ax.secondary_xaxis('top', functions=(lambda x1: shift-x1,)*2)
ax2.set_xlabel('Average horiz. vel.')
ax.set_xlim(0, shift)


for folder in folders:
    dataFile = f"{folder}/{output}"
    if not os.path.isfile(dataFile):
        files = OutputFiles(folder)
        zFine, uMeanFine, bRMSFine, deltaU, deltaT = files.getLayersQuantities(0)

        zFine, uMeanFine, bRMSFine, _, _ = files.getLayersQuantities(
            0, iBeg=iBeg, iEnd=iEnd, step=step, verbose=verbose)
        if verbose: print(f" -- saving profiles into {dataFile}...")
        profiles = np.array([zFine, uMeanFine, bRMSFine])
        np.savetxt(dataFile, profiles, header="spectrum[uv,z]")
    else:
        if verbose: print(f" -- loading profiles from {dataFile} ...")
        zFine, uMeanFine, bRMSFine = np.loadtxt(dataFile)

    if verbose: print(" -- ploting and saving figure to file ...")

    nFine = len(zFine)
    deltaU = zFine[np.argmax(uMeanFine[:nFine//2])]
    deltaT = zFine[np.argmax(bRMSFine[:nFine//2])]

    p = ax.plot(bRMSFine, zFine, label="$T_{rms}$"f" ({folder})")
    ax.plot(shift-uMeanFine, zFine, "--", c=p[0].get_color(), label="$U_{mean}$"f" ({folder})")

    ax.hlines(deltaU, 0, shift, linestyles=":", colors="black")
    ax.hlines(1-deltaU, 0, shift, linestyles=":", colors="black")
    ax.hlines(1-deltaT, 0, shift, linestyles=":", colors="black")
    ax.hlines(deltaT, 0, shift, linestyles=":", colors="black")

plt.legend()
plt.tight_layout()
plt.savefig(os.path.splitext(output)[0]+".pdf")
if verbose: print(" -- done !")
