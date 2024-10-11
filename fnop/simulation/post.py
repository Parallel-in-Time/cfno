import h5py
import glob
import random
import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class OutputFiles():
    """
    Object to load and manipulate hdf5 Dedalus generated solution output
    """
    def __init__(self, folder, inference=False):
        self.folder = folder
        self.inference = inference
        fileNames = glob.glob(f"{self.folder}/*.h5")
        fileNames.sort(key=lambda f: int(f.split("_s")[-1].split(".h5")[0]))
        self.files = fileNames
        self._iFile = None
        vData0 = self.file(0)['tasks']['velocity']
        if self.inference:
             self.x = np.array(vData0[0,0,:,0])
             self.z = np.array(vData0[0,1,0,:])
             print(f'x-grid: {self.x.shape}')
             print(f'z-grid: {self.z.shape}')
             print(f'timesteps: {np.array(vData0[:,0,0,0]).shape}')
        else:
            self.x = np.array(vData0.dims[2]["x"])
            self.z = np.array(vData0.dims[3]["z"])


    def file(self, iFile:int):
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

    def vData(self, iFile:int):
        return self.file(iFile)['tasks']['velocity']

    def bData(self, iFile:int):
        return self.file(iFile)['tasks']['buoyancy']

    def pData(self, iFile:int):
        return self.file(iFile)['tasks']['pressure']

    def times(self, iFile:int):
        if self.inference:
            return np.array(self.vData(iFile)[:,0,0,0])
        else:
            return np.array(self.vData(iFile).dims[0]["sim_time"])

    def nTimes(self, iFile:int):
        return self.times(iFile).size

    def getMeanProfiles(self, iFile:int, buoyancy=False, pressure=False):
        """_summary_

        Args:
            iFile (int): file index
            buoyancy (bool, optional): return buoyancy profile. Defaults to False.
            pressure (bool, optional): return pressure profile. Defaults to False.

        Returns:
           profilr (list): mean profiles of velocity, buoyancy and pressure
        """
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

    def getMeanSpectrum(self, iFile:int):
        """
        Function to get mean spectrum
        Args:
            iFile (int): file index

        Returns:
            energy_spectrum (list): mean energy spectrum
        """
        energy_spectrum = []
        for i in range(2):
            u = self.vData(iFile)[:, i]                    #(time_index, Nx, Nz)
            spectrum = np.fft.rfft(u, axis=-2)             # over Nx -->  #(time_index, k, Nz)
            spectrum *= np.conj(spectrum)                  #(time_index, k, Nz)
            spectrum /= spectrum.shape[-2]                 # normalize with Nx --> (time_index, k, Nz)
            spectrum = np.mean(spectrum.real, axis=-1)     # mean over Nz --> (time_index,k)
            energy_spectrum.append(spectrum)
        return energy_spectrum                             # (2,time_index,k)

    def getFullMeanSpectrum(self, iBeg:int, iEnd=None):
        """
        Function to get full mean spectrum

        Args:
            iBeg (int): starting file index
            iEnd (int, optional): stopping file index. Defaults to None.

        Returns:
           sMean (np.ndarray): mean spectrum
           k (np.ndarray): wave number
        """
        if iEnd is None:
            iEnd = self.nFiles
        sMean = []
        for iFile in range(iBeg, iEnd):
            energy_spectrum = self.getMeanSpectrum(iFile)
            sx, sz = energy_spectrum[0], energy_spectrum[1]      # (1,time_index,k)
            sMean.append(np.mean((sx+sz)/2, axis=0))             # mean over time ---> (2, k)
        sMean = np.mean(sMean, axis=0)                           # mean over x and z ---> (k)
        np.savetxt(f'{self.folder}/spectrum.txt', np.vstack((sMean, self.k)))
        return sMean, self.k

def checkDNS(sMean:np.ndarray, k:np.ndarray, vRatio:int=4, nThrow:int=1):
    """
    Funciton to check DNS
    Args:
        sMean (np.ndarray): mean spectrum
        k (np.ndarray): wave number
        vRatio (int, optional): #to-do
        nThrow (int, optional): number of values to exclude fitting. Defaults to 1.

    Returns:
        status (bool): if under-resolved or not
        [a, b, c] (float): polynomial coefficients
        x, y (float): variable values
        nValues (int): # to-do
    """
    nValues = k.size//vRatio

    y = np.log(sMean[-nValues-nThrow:-nThrow])
    x = np.log(k[-nValues-nThrow:-nThrow])

    def fun(coeffs):
        a, b, c = coeffs
        return np.linalg.norm(y - a*x**2 - b*x - c)

    res = sco.minimize(fun, [0, 0, 0])
    a, b, c = res.x
    status = "under-resolved" if a > 0 else "DNS !"

    return status, [a, b, c], x, y, nValues

def generateChunkPairs(folder:str, N:int, M:int,
                       tStep:int=1, xStep:int=1, zStep:int=1,
                       shuffleSeed=None
):
    """
    Function to generate chunk pairs

    Args:
        folder (str): path to dedalus hdf5 data
        N (int): timesteps of dt
        M (int): size of chunk
        tStep (int, optional): time slicing. Defaults to 1.
        xStep (int, optional): x-grid slicing. Defaults to 1.
        zStep (int, optional): z-grid slicing. Defaults to 1.
        shuffleSeed (int, optional): seed for random shuffle. Defaults to None.

    Returns:
        pairs (list): chunk pairs
    """
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
        chunk2 = np.stack((vxData[i+N:i+N+M],vzData[i+N:i+N+M], bData[i+N:i+N+M],
                           pData[i+N:i+N+M]), axis=1)
        # chunks are shape (M, 4, Nx//xStep, Nz//zStep)
        assert chunk1.shape == chunk2.shape
        pairs.append((chunk1, chunk2))

    # shuffle if a seed is given
    if shuffleSeed is not None:
        random.seed(shuffleSeed)
        random.shuffle(pairs)

    return pairs


def contourPlot(field, x, y,
                title=None, refField=None, refTitle=None, saveFig=False,
                closeFig=True):

    fig, axs = plt.subplots(1 if refField is None else 2)
    ax = axs if refField is None else axs[0]

    def setup(ax):
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("x")
        ax.set_ylabel("z")

    def setColorbar(im, ax):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

    im = ax.pcolormesh(x, y, field)
    setColorbar(im, ax)
    ax.set_title(title)
    setup(ax)

    if refField is not None:
        im = axs[1].pcolormesh(x, y, refField)
        setColorbar(im, axs[1])
        axs[1].set_title(refTitle)
        setup(axs[1])

    plt.tight_layout()
    if saveFig:
        plt.savefig(saveFig)
    if closeFig:
        plt.close(fig)
