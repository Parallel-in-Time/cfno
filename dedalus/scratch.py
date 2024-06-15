#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import h5py
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco

class OutputFiles():

    def __init__(self, folder):
        fileNames = glob.glob(f"{folder}/*.h5")
        fileNames.sort(key=lambda f: int(f.split("_s")[-1].split(".h5")[0]))
        self.files = fileNames[:-1]
        self._iFile = None
        vData0 = self.file(0)['tasks']['velocity']

        self.x = np.array(vData0.dims[2]["x"])
        self.z = np.array(vData0.dims[3]["z"])

    def file(self, iFile):
        if iFile != self._iFile:
            self._iFile = iFile
            self._file = h5py.File(self.files[iFile], mode='r')
        return self._file

    @property
    def nFiles(self):
        return len(self.files)

    @property
    def nX(self):
        return self.x.size

    @property
    def nZ(self):
        return self.z.size

    @property
    def k(self):
        nX = self.nX
        k = np.fft.rfftfreq(nX, 1/nX) + 0.5
        return k

    def vData(self, iFile):
        return self.file(iFile)['tasks']['velocity']

    def bData(self, iFile):
        return self.file(iFile)['tasks']['buoyancy']

    def times(self, iFile):
        return np.array(self.vData(iFile).dims[0]["sim_time"])

    def nTimes(self, iFile):
        return self.times(iFile).size

    def getMeanProfiles(self, iFile):
        out = []
        for i in range(2):
            u = self.vData(iFile)[:, i]
            mean = np.mean(abs(u), axis=1)
            out.append(mean)
        return out

    def getMeanSpectrum(self, iFile):
        out = []
        for i in range(2):
            u = self.vData(iFile)[:, i]
            spectrum = np.fft.rfft(u, axis=-2)
            spectrum *= np.conj(spectrum)
            spectrum /= spectrum.shape[-2]
            spectrum = np.mean(spectrum.real, axis=-1)
            out.append(spectrum)
        return out

    def getFullMeanSpectrum(self, iBeg=0, iEnd=None):
        if iEnd is None:
            iEnd = self.nFiles
        k = self.k
        sMean = []
        for iFile in range(iBeg, iEnd):
            sx, sz = self.getMeanSpectrum(iFile)
            sMean.append(np.mean((sx+sz)/2, axis=0))
        sMean = np.mean(sMean, axis=0)
        return sMean, k


out = OutputFiles("snapshots")
writeFig = True
zRel = out.z - 0.5

if writeFig:

    os.makedirs("spectrum", exist_ok=True)
    k = out.k
    sMean = []
    for iFile in range(10, out.nFiles):
        sx, sz = out.getMeanSpectrum(iFile)
        sMean.append(np.mean((sx+sz)/2, axis=0))
    sMean = np.mean(sMean, axis=0)
    # np.round(sMean, decimals=14, out=sMean)

    plt.figure("spectrum")
    plt.loglog(k[:-1], sMean[:-1], label="Emean")
    plt.xlabel("wavenumber")
    plt.ylabel("Energy spectrum")
    plt.ylim(1e-20, 1e4)
    plt.legend(loc="upper right")
    plt.tight_layout()


nValues = k.size//4
nThrown = 1

y = np.log(sMean[-nValues-nThrown:-nThrown])
x = np.log(k[-nValues-nThrown:-nThrown])

def fun(coeffs):
    a, b, c = coeffs
    return np.linalg.norm(y - a*x**2 - b*x - c)

res = sco.minimize(fun, [0, 0, 0])

a, b, c = res.x
y = a*x**2 + b*x + c
x = np.exp(x)
y = np.exp(y)
plt.plot(x, y)

status = "under-resolved" if a > 0 else "DNS !"
print(f"{status} (a={a})")


        # for i, t in enumerate(out.times(iFile)):
        #     print(f"spectrum plot {idx}")
        #     plt.figure("spectrum").clear()
        #     plt.semilogy(k, sx[i], label="Ex")
        #     plt.semilogy(k, sz[i], label="Ez")
        #     plt.xlabel("wavenumber")
        #     plt.ylabel("E")
        #     plt.ylim(1e-20, 1e4)
        #     plt.title(f"t={t:1.2f}s")
        #     plt.legend(loc="upper right")
        #     plt.tight_layout()
        #     plt.savefig(f"spectrum/fig_{idx:04d}.png", bbox_inches="tight")
        #     idx += 1

if False:
    os.makedirs("profiles", exist_ok=True)
    idx = 0
    for iFile in range(out.nFiles):
        uxMean, uzMean = out.getMeanProfiles(iFile)
        for i, t in enumerate(out.times(iFile)):
            print(f"profiles plot {idx}")
            fig = plt.figure("profiles")
            fig.clear()
            plt.plot(uxMean[i], zRel, label="uxMean")
            plt.plot(uzMean[i], zRel, label="uzMean")
            plt.xlabel("velocity")
            plt.xlim(0, 0.35)
            plt.ylabel("z")
            plt.title(f"t={t:1.2f}s")
            plt.legend(loc="center")
            plt.tight_layout()
            plt.savefig(f"profiles/fig_{idx:04d}.png", bbox_inches="tight")
            idx += 1

    os.makedirs("buyoancy", exist_ok=True)
    idx = 0
    for iFile in range(out.nFiles):
        bMean = np.mean(out.bData(iFile), axis=-2)
        for i, t in enumerate(out.times(iFile)):
            print(f"buyoancy plot {idx}")
            fig = plt.figure("buyoancy")
            fig.clear()
            plt.plot(bMean[i], zRel, label="bMean")
            plt.xlabel("buyoancy")
            # plt.xlim(0, 0.35)
            plt.ylabel("z")
            plt.title(f"t={t:1.2f}s")
            plt.legend(loc="center")
            plt.tight_layout()
            plt.savefig(f"buyoancy/fig_{idx:04d}.png", bbox_inches="tight")
            idx += 1
