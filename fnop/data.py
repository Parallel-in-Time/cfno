"""
Minimal utilities to generate training and validation data
"""
import glob

import h5py
import numpy as np
import torch as th
from torch.utils.data import Dataset, DataLoader, random_split, Subset

from fnop.simulation.post import OutputFiles


class HDF5Dataset(Dataset):

    def __init__(self, dataFile):
        self.file = h5py.File(dataFile, 'r')
        self.inputs = self.file['inputs']
        self.outputs = self.file['outputs']
        assert len(self.inputs) == len(self.outputs), \
            f"different sample number for inputs and outputs ({len(self.inputs)},{len(self.outputs)})"

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inpt, outp = self.sample(idx)
        return th.tensor(inpt), th.tensor(outp)

    def __del__(self):
        try:
            self.file.close()
        except:
            pass

    def sample(self, idx):
        return self.inputs[idx], self.outputs[idx]

    @property
    def infos(self):
        return self.file["infos"]

    @property
    def grid(self):
        return self.infos["xGrid"][:], self.infos["yGrid"][:]

    @property
    def outType(self):
        return self.infos["outType"][()].decode("utf-8")

    @property
    def outScaling(self):
        return float(self.infos["outScaling"][()])

    def printInfos(self):
        xGrid, yGrid = self.grid
        infos = self.infos
        print(f" -- grid shape : ({xGrid.size}, {yGrid.size})")
        print(f" -- grid domain : [{xGrid.min():.1f}, {xGrid.max():.1f}] x [{yGrid.min():.1f}, {yGrid.max():.1f}]")
        print(f" -- nSimu : {infos['nSimu'][()]}")
        print(f" -- dtData : {infos['dtData'][()]:1.2g}")
        print(f" -- inSize : {infos['inSize'][()]}")
        print(f" -- outStep : {infos['outStep'][()]}")
        print(f" -- inStep : {infos['inStep'][()]}")
        print(f" -- nSamples (per simu) : {infos['nSamples'][()]}")
        print(f" -- nSamples (total) : {infos['nSamples'][()]*infos['nSimu'][()]}")
        print(f" -- dtInput : {infos['dtInput'][()]:1.2g}")
        print(f" -- outType : {infos['outType'][()].decode('utf-8')}")
        print(f" -- outScaling : {infos['outScaling'][()]:1.2g}")


def createDataset(dataDir, inSize, outStep, inStep, outType, outScaling, dataFile):
    assert inSize == 1, "inSize != 1 not implemented yet ..."
    simDirs = glob.glob(f"{dataDir}/simu_*")
    nSimu = len(simDirs)

    # -- retrieve informations from first simulation
    outFiles = OutputFiles(f"{simDirs[0]}/run_data")

    times = outFiles.times().ravel()
    dtData = times[1]-times[0]
    dtInput = dtData*outStep  # noqa: F841 (used lated by an eval call)
    xGrid, yGrid = outFiles.x, outFiles.y  # noqa: F841 (used lated by an eval call)

    nFields = sum(outFiles.nFields)
    sRange = range(0, nFields-inSize-outStep+1, inStep)
    nSamples = len(sRange)

    print(f"Creating dataset from {len(simDirs)} simulations, {nSamples} samples each ...")
    dataset = h5py.File(dataFile, "w")
    for name in ["inSize", "outStep", "inStep", "outType", "outScaling",
                 "dtData", "dtInput", "xGrid", "yGrid",
                 "nSimu", "nSamples"]:
        try:
            dataset.create_dataset(f"infos/{name}", data=np.asarray(eval(name)))
        except:
            dataset.create_dataset(f"infos/{name}", data=eval(name))

    dataShape = (nSamples*nSimu, *outFiles.shape)
    inputs = dataset.create_dataset("inputs", dataShape)
    outputs = dataset.create_dataset("outputs", dataShape)
    for iSim, dataDir in enumerate(simDirs):
        outFiles = OutputFiles(f"{dataDir}/run_data")
        print(f" -- sampling data from {outFiles.folder}")
        for iSample, iField in enumerate(sRange):
            inpt, outp = outFiles.fields(iField), outFiles.fields(iField+outStep)
            if outType == "update":
                outp -= inpt
                if outScaling != 1:
                    outp *= outScaling
            inputs[iSim*nSamples + iSample] = inpt
            outputs[iSim*nSamples + iSample] = outp
    print(" -- done !")


def getDataLoaders(dataFile, trainRatio=0.8, batchSize=20, seed=None):
    dataset = HDF5Dataset(dataFile)
    nBatches = len(dataset)
    trainSize = int(trainRatio*nBatches)
    valSize = nBatches - trainSize

    if seed is None:
        trainIdx = list(range(0, trainSize))
        valIdx = list(range(trainSize, nBatches))
        trainSet = Subset(dataset, trainIdx)
        valSet = Subset(dataset, valIdx)
    else:
        generator = th.Generator().manual_seed(seed)
        trainSet, valSet = random_split(
            dataset, [trainSize, valSize], generator=generator)

    trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True)
    valLoader = DataLoader(valSet, batch_size=batchSize, shuffle=False)

    return trainLoader, valLoader, dataset
