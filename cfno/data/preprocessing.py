"""
Minimal utilities to generate training and validation data
"""

import h5py
import torch
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from cfno.simulation.post import OutputFiles

class HDF5Dataset(Dataset):

    def __init__(self, dataFile, **kwargs):
        """
        Dataset reader and getitem for DataLoader

        Args:
            dataFile (hdf5): data file 
            
        """
        
        self.file = h5py.File(dataFile, 'r')
        self.inputs = self.file['inputs']
        self.outputs = self.file['outputs']
        xGrid, yGrid = self.grid
        self.nX = xGrid.size
        self.nY = yGrid.size
        self.kX = kwargs.get('kX', 12)
        self.kY = kwargs.get('kY', 12)
 
        assert len(self.inputs) == len(self.outputs), \
            f"different sample number for inputs and outputs ({len(self.inputs)},{len(self.outputs)})"
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inpt, outp = self.sample(idx)
        return torch.tensor(inpt), torch.tensor(outp)

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

    def calc_sliceMin(self, n, modes):
        """
        Finding min number of points to satisfy
        n/2 + 1 >= fourier modes
        """
        slice_min = 2*(modes-1)
        if slice_min < n:
            return slice_min
        else:
            print("Insufficient number of points to slice")
            return 0

    def printInfos(self):
        xGrid, yGrid = self.grid
        infos = self.infos
        print(f" -- grid shape : ({xGrid.size}, {yGrid.size})")
        print(f" -- grid domain : [{xGrid.min():.1f}, {xGrid.max():.1f}] x [{yGrid.min():.1f}, {yGrid.max():.1f}]")
        print(f" -- nSimu : {infos['nSimu'][()]}")
        print(f" -- dtData : {infos['dtData'][()]:1.2g}")
        print(f" -- inSize : {infos['inSize'][()]}")                # T_in
        print(f" -- outStep : {infos['outStep'][()]}")              # T
        print(f" -- inStep : {infos['inStep'][()]}")                # tStep
        print(f" -- nSamples (per simu) : {infos['nSamples'][()]}")
        print(f" -- nSamples (total) : {infos['nSamples'][()]*infos['nSimu'][()]}")
        print(f" -- dtInput : {infos['dtInput'][()]:1.2g}")
        print(f" -- outType : {infos['outType'][()].decode('utf-8')}")
        print(f" -- outScaling : {infos['outScaling'][()]:1.2g}")
 
def createDataset(
        dataDir, inSize, outStep, inStep, outType, outScaling, dataFile,
        dryRun=False, verbose=False, pySDC=False, **kwargs):

    assert inSize == 1, "inSize != 1 not implemented yet ..."
    simDirsSorted = sorted(glob.glob(f"{dataDir}/simu_*"), key=lambda f: int(f.split('simu_',1)[1]))
    nSimu = int(kwargs.get("nSimu", len(simDirsSorted)))
    simDirs = simDirsSorted[:nSimu]
    print('Using Simulations:')
    for s in simDirs:
        print(f" -- {s}")

    # -- retrieve informations from first simulation
    if pySDC:
        from pySDC.helpers.fieldsIO import FieldsIO
        outFiles = FieldsIO.fromFile(f"{simDirs[0]}/run_data/outputs.pysdc")
        nFields = outFiles.nFields
        fieldShape = (outFiles.nVar, outFiles.nX, outFiles.nY)
        times = outFiles.times
        xGrid, yGrid = outFiles.header["coordX"], outFiles.header["coordY"]  # noqa: F841 (used lated by an eval call)
    else:
        outFiles = OutputFiles(f"{simDirs[0]}/run_data")
        nFields = sum(outFiles.nFields)
        fieldShape = outFiles.shape
        times = outFiles.times().ravel()
        xGrid, yGrid = outFiles.x, outFiles.y  # noqa: F841 (used lated by an eval call)

    dtData = times[1]-times[0]
    dtInput = dtData*outStep  # noqa: F841 (used lated by an eval call)
    dtSample = dtData*inStep  # noqa: F841 (used lated by an eval call)

    iBeg = int(kwargs.get("iBeg", 0))
    iEnd = int(kwargs.get("iEnd", nFields))
    sRange = range(iBeg, iEnd-inSize-outStep+1, inStep)
    nSamples = len(sRange)
    print(f'selector: {sRange},  outStep: {outStep}, inStep: {inStep}, iBeg: {iBeg}, iEnd: {iEnd}')

    infoParams = [
        "inSize", "outStep", "inStep", "outType", "outScaling", "iBeg", "iEnd",
        "dtData", "dtInput", "xGrid", "yGrid", "nSimu", "nSamples", "dtSample",
    ]

    if dryRun:
        print(f"To create : dataset from {nSimu} simulations, {nSamples} samples each ...")
        for name in infoParams:
            if "Grid" not in name:
                print(f" -- {name} : {eval(name)}")
        return

    print(f"Creating dataset from {nSimu} simulations, {nSamples} samples each ...")
    dataset = h5py.File(dataFile, "w")
    for name in infoParams:
        try:
            dataset.create_dataset(f"infos/{name}", data=np.asarray(eval(name)))
        except:
            dataset.create_dataset(f"infos/{name}", data=eval(name))

    dataShape = (nSamples*nSimu, *fieldShape)
    print(f'data shape: {dataShape}')
    inputs = dataset.create_dataset("inputs", dataShape)
    outputs = dataset.create_dataset("outputs", dataShape)
    for iSim, dataDir in enumerate(simDirs):
        if pySDC:
            outFiles = FieldsIO.fromFile(f"{dataDir}/run_data/outputs.pysdc")
        else:
            outFiles = OutputFiles(f"{dataDir}/run_data")
        print(f" -- sampling data from {dataDir}/run_data")
        for iSample, iField in enumerate(sRange):
            if verbose:
                print(f"\t -- creating sample {iSample+1}/{nSamples}")
            if pySDC:
                inpt, outp = outFiles.readField(iField)[-1], outFiles.readField(iField+outStep)[-1]
            else:
                inpt, outp = outFiles.fields(iField), outFiles.fields(iField+outStep).copy()
            if outType == "update":
                outp -= inpt
                if outScaling != 1:
                    outp *= outScaling
            inputs[iSim*nSamples + iSample] = inpt
            outputs[iSim*nSamples + iSample] = outp
    dataset.close()
    print(" -- done !")

def getDataLoaders(dataFile, 
                   trainRatio=0.8, 
                   batchSize=20,
                   seed=None, 
                   **kwargs):

    
    dataset = HDF5Dataset(dataFile)

    dataset.printInfos()
    nBatches = len(dataset)
    trainSize = int(trainRatio*nBatches)
    valSize = nBatches - trainSize

    if seed is None:
        trainIdx = list(range(0, trainSize))
        valIdx = list(range(trainSize, nBatches))
        trainSet = Subset(dataset, trainIdx)
        valSet = Subset(dataset, valIdx)
    else:
        generator = torch.Generator().manual_seed(seed)
        trainSet, valSet = random_split(
            dataset, [trainSize, valSize], generator=generator)

    trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True, num_workers=0, pin_memory=True)
    valLoader = DataLoader(valSet, batch_size=batchSize, shuffle=False, num_workers=0, pin_memory=True)

    return trainLoader, valLoader, dataset

