#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Plot Spectrum and Profiles for 2D Rayleigh Benard Convection

Usage:
    python data_processing.py --dir_name <data_dir> 
    
Options:
    --spectrum_plot : for plotting energy spectrum
    --profile_plot  : for plotting profiles
    --checkDNS_plot : for plotting second order polynomial fit on energy spectrum
    
Comment:
    Plots are saved in `dir_name/plots`
"""


import os
import h5py
import glob
import random
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import scipy.optimize as sco


class OutputFiles():

    def __init__(self, folder):
        self.folder = folder
        fileNames = glob.glob(f"{self.folder}/*.h5")
        fileNames.sort(key=lambda f: int(f.split("_s")[-1].split(".h5")[0]))
        self.files = fileNames
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
    
    def pData(self, iFile):
        return self.file(iFile)['tasks']['pressure']

    def times(self, iFile):
        return np.array(self.vData(iFile).dims[0]["sim_time"])

    def nTimes(self, iFile):
        return self.times(iFile).size

    def getMeanProfiles(self, iFile, buoyancy=False, pressure=False):
        profile = []
        for i in range(2):                              # x and z components (time_index, 2, Nx, Nz)
            u = self.vData(iFile)[:, i]                 # (time_index, Nx, Nz)
            mean = np.mean(abs(u), axis=1)              # avg over Nx ---> (time_index, Nz)
            profile.append(mean)                        # (2, time_index, Nz)
        if buoyancy:
            b = self.bData(iFile)                       #(time_index, Nx, Nz)
            profile.append(np.mean(b, axis=1))          # (time_index, Nz)
        if pressure:
            p = self.pData(iFile)                       #(time_index, Nx, Nz)
            profile.append(np.mean(p, axis=1))          # (time_index, Nz)
        return profile

    def getMeanSpectrum(self, iFile):
        energy_spectrum = []
        for i in range(2):
            u = self.vData(iFile)[:, i]                #(time_index, Nx, Nz)
            spectrum = np.fft.rfft(u, axis=-2)         # over Nx -->  #(time_index, k, Nz)
            spectrum *= np.conj(spectrum)              #(time_index, k, Nz)
            spectrum /= spectrum.shape[-2]                 # normalize with Nx --> (time_index, k, Nz)
            spectrum = np.mean(spectrum.real, axis=-1)     # mean over Nz --> (time_index,k)
            energy_spectrum.append(spectrum)
        return energy_spectrum                         # (2,time_index,k)

    def getFullMeanSpectrum(self, iBeg, iEnd=None):
        if iEnd is None:
            iEnd = self.nFiles
        sMean = []
        for iFile in range(iBeg, iEnd):
            sx, sz = self.getMeanSpectrum(iFile)        # (1,time_index,k)
            sMean.append(np.mean((sx+sz)/2, axis=0))    # mean over time ---> (2, k)
        sMean = np.mean(sMean, axis=0)                  # mean over x and z ---> (k)
        np.savetxt(f'{self.folder}/spectrum.txt', np.vstack((sMean, self.k)))
        return sMean, self.k
    
    def rbc_data(self, filename, time, tasks=False, scales=False):
        with h5py.File(filename, mode="r") as f:
            b_t = f["tasks/buoyancy"][time]
            vel_t = f["tasks/velocity"][time]
            p_t = f["tasks/pressure"][time]
            iteration = f["scales/iteration"][time]
            sim_time  = f["scales/sim_time"][time]
            time_step = f["scales/timestep"][time]
            wall_time = f["scales/wall_time"][time]
            write_no = f["scales/write_number"][time]
            
        f.close()
        if tasks and scales:
             return vel_t,b_t, p_t, write_no, iteration, sim_time, time_step, wall_time
        elif tasks:
             return vel_t,b_t, p_t
        elif scales:
             return write_no, iteration, sim_time, time_step, wall_time
        else:
             raise ValueError("Nothing to return!")
    
    def data_process(self, xStep=1, zStep=1):
        index = 0
        inputs = []
        filename = f'{processed_data}/input_data.h5'    # TODO: don't use a global variable here ...
        with h5py.File(filename, "w") as data:
            for i,file in enumerate(self.files):
                iter_no = self.times(i).shape[0]
                # print(i, file, iter_no)
                for t in range(iter_no):
                    vel_t,b_t, p_t, write_no, iteration, sim_time, time_step, wall_time = self.rbc_data(file, t, True, True)
                    inputs.append(np.concatenate((vel_t[0,::xStep,::zStep], vel_t[1,::xStep,::zStep], b_t, p_t), axis = 0))
                    index = index + 1
            data['input'] = inputs
            data.close()
            return np.array(inputs)
    

def checkDNS(sMean, k, vRatio=4, nThrow=1):
    nValues = k.size//vRatio
    nThrow = nThrow

    y = np.log(sMean[-nValues-nThrow:-nThrow])
    x = np.log(k[-nValues-nThrow:-nThrow])

    def fun(coeffs):
        a, b, c = coeffs
        return np.linalg.norm(y - a*x**2 - b*x - c)

    res = sco.minimize(fun, [0, 0, 0])
    a, b, c = res.x
    status = "under-resolved" if a > 0 else "DNS !"

    return status, [a, b, c], x, y, nValues

def generateChunkPairs(folder, N, M, tStep=1, xStep=1, zStep=1, shuffleSeed=None):
    out = OutputFiles(folder)

    pairs = []
    vxData, vzData,  bData, pData = [], [],  [], []
    for iFile in range(0, out.nFiles):
        vxData.append(out.vData(iFile)[:, 0, ::xStep, ::zStep])
        vzData.append(out.vData(iFile)[:, 1, ::xStep, ::zStep])
        bData.append(out.bData(iFile)[:, ::xStep, ::zStep])
        pData.append(out.pData(iFile)[:, ::xStep, ::zStep])
    # stack all arrays
    vxData = np.concatenate(vxData)
    vzData = np.concatenate(vzData)
    bData = np.concatenate(bData)
    pData = np.concatenate(pData)
    
    assert vxData.shape[0] == vzData.shape[0]
    assert vzData.shape[0] == pData.shape[0]
    assert vzData.shape[0] == bData.shape[0]
    nTimes = vxData.shape[0]

    for i in range(0, nTimes-M-N+1, tStep):
        chunk1 = np.stack((vxData[i:i+M],vzData[i:i+M],bData[i:i+M],pData[i:i+M]), axis=1)
        chunk2 = np.stack((vxData[i+N:i+N+M],vzData[i+N:i+N+M], bData[i+N:i+N+M], pData[i+N:i+N+M]), axis=1)
        # chunks are shape (M, 4, Nx//xStep, Nz//zStep)
        assert chunk1.shape == chunk2.shape
        pairs.append((chunk1, chunk2))

    # shuffle if a seed is given
    if shuffleSeed is not None:
        random.seed(shuffleSeed)
        random.shuffle(pairs)

    return pairs


def plotting(args_data):
    
    dirName = args_data.dir_name
    plot_path = Path(f'{dirName}/plots')
    plot_path.mkdir(parents=True, exist_ok=True)

    processed_data = Path(f'{dirName}/processed_data')
    processed_data.mkdir(parents=True, exist_ok=True)
    info_file = Path(f"{dirName}/00_infoSimu.txt")

    if info_file.is_file():
        with open(info_file, "r") as f:
            infos = f.read().strip()
        infos = {val.split(" : ")[0]: val.split(" : ")[1] for val in infos.split('\n')}
        infos["Rayleigh"] = float(infos["Rayleigh"])
        infos["Nx"], infos["Nz"] = [int(n) for n in infos["Nx, Nz"].split(", ")]
    else:
        infos["Rayleigh"] = 1e-7
        infos["Nx"] = 256
        infos["Nz"] = 64
    print(f'Setting RayleighNumber={infos["Rayleigh"]}, Nx={infos["Nx"]} and Nz={infos["Nz"]}')
        
    out = OutputFiles(dirName)
    sMean, k = out.getFullMeanSpectrum(0)
    sMean /= infos["Nx"]
    # sMean /= np.prod(infos["Nx, Nz"])

    # Spectrum 
    if args_data.spectrum_plot: 
        plt.figure("spectrum")
        plt.title(fr"Mean Energy Spectrum vs Wave Number on {infos['Nx']} $\times$ {infos['Nz']} grid")
        plt.xlabel("Wavenumber")
        plt.ylabel("Mean Energy Spectrum")
        plt.grid()
        plt.loglog(k[:-1], sMean[:-1], label=f"Ra={infos['Rayleigh']:1.1e}")
        plt.savefig(f"{dirName}/plots/spectrum.pdf")

    ## checkDNS
    if args_data.checkDNS_plot:
        vRatio = 4
        nThrow = 3
        status, (a, b, c), x, y, nValues = checkDNS(sMean, k, vRatio, nThrow)
        print(f"{status} (a={a})")
        y1 = a*x**2 + b*x + c
        x = np.exp(x)
        y1 = np.exp(y1)
        plt.title("Spectrum Fitting with second order polynomial")
        plt.plot(x, np.exp(y), 'k--', label="spectrum")
        plt.plot(x,y1,'r*', label=r"fit")
        plt.legend()
        plt.grid(True)
        plt.xlabel("Wavenumber")
        plt.ylabel("Spectrum")
        plt.savefig(f"{dirName}/plots/polyfit.pdf")

    # Profiles
    if args_data.profile_plot:
        uxMean, uzMean, bVals, pVals = [], [], [], []
        time = []
        for i in range(out.nFiles):
            ux, uz, b, p = out.getMeanProfiles(i, buoyancy=True, pressure=True)
            time += list(out.times(i))
            uxMean.append(ux)
            uzMean.append(uz)
            bVals.append(b)
            pVals.append(p)
        time = np.array(time)
        uxMean = np.concatenate(uxMean)
        uzMean = np.concatenate(uzMean)
        bVals = np.concatenate(bVals)
        pVals = np.concatenate(pVals)

        plt.figure("mid-profile")
        plt.title("Mid-profile mean velocity vs Time")
        plt.plot(time, uxMean[:,  infos["Nz"]//2], label=r"$\bar{u}_x$")
        plt.plot(time, uzMean[:,  infos["Nz"]//2], label=r"$\bar{u}_z$")
        plt.legend()
        plt.grid(True)
        plt.xlabel("Time")
        plt.ylabel("Mid-profile value")
        plt.savefig(f"{dirName}/plots/mid_profile.pdf")

        plt.figure("profiles")
        plt.title(" Profile in Z-coordinate")
        plt.plot(uxMean[500:].mean(axis=0), out.z, label=r"$\bar{u}_x$")
        plt.plot(uzMean[500:].mean(axis=0), out.z, label=r"$\bar{u}_z$")
        plt.plot(bVals[500:].mean(axis=0), out.z, label=r"$\bar{b}$")
        plt.plot(pVals[500:].mean(axis=0),out.z, label=r"$\bar{p}$")
        plt.legend()
        plt.grid(True)
        plt.xlabel("Profile")
        plt.ylabel("z-coordinate")
        plt.savefig(f"{dirName}/plots/profile.pdf")
        
    
if __name__ == '__main__':
    parser_data = argparse.ArgumentParser(description='Data Analysis')
    parser_data.add_argument('--dir_name', type=str,
                        help="Folder name to store data")
    parser_data.add_argument('--spectrum_plot', action='store_true',
                        help='plot spectrum vs k')
    parser_data.add_argument('--checkDNS_plot', action='store_true',
                        help='check second order polynomial fit')
    parser_data.add_argument('--profile_plot', action='store_true',
                        help='plotting z-coord profile for \
                            velocity, buoyancy and pressure')
    args_data, unknown = parser_data.parse_known_args()
    plotting(args_data)
 