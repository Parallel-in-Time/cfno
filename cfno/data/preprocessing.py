"""
Minimal utilities to generate training and validation data
"""

import h5py
import torch
import glob
import numpy as np
from cfno.utils import UnitGaussianNormalizer

from torch.utils.data import Dataset, DataLoader, random_split, Subset, DistributedSampler
from cfno.simulation.post import OutputFiles
from cfno.communication import get_local_rank, get_world_size

class FNOData():
    """
    Processing Dedalus data
    """
    def __init__(self,
                 nx:int,
                 ny:int,
                 dt:float,
                 dim:str,
                 start_time:float,
                 stop_time:float,
                 xStep:int=1,
                 yStep:int=1,
                 tStep:int=1,
                 timestep:int=1,
                 T_in:int=1,
                 T:int=1,
                 **kwargs
    ):
        """

        Args:
            nx (int): size of nx
            ny (int): size of ny
            dt (float): delta timestep 
            dim (str): FNO2D or FNO3D strategy
            start_time (float): start time
            stop_time (float): stop time
            xStep (int): slicing for x. Defaults to 1.
            yStep (int): slicing for y. Defaults to 1.
            tStep (int): time slice. Defaults to 1.
            timestep (int): time interval. Defaults to 1.
            T_in (int):number of input timesteps. Defaults to 1.
            T (int): number of output timesteps. Defaults to 1.
        """
        super().__init__()
        self.dim = dim
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.xStep = xStep
        self.yStep = yStep
        self.tStep = tStep
        self.T_in = T_in
        self.T = T
        self.start_time = start_time
        self.stop_time = stop_time
        self.timestep = timestep
        self.nx_state = 4*self.nx  # stacking [velx,velz,buoyancy,pressure]
  
    def get_concat_data(self, task:str, nsamples:int, reader, multistep:bool=True):
        """
        Data for FNO model

        Args:
            task (str): 'train', or 'val' or 'test
            nsamples (int): number of simulation samples
            reader: hdf5 file reader
            multistep (bool): load multiple time index data. Defaults to True.

        Returns:
            inputs (torch.tensor): input for FNO
            outputs (torch.tensor): output for FNO
            
        """
      
      
        print(f'{task} data: {reader[task].shape}')
        # [samples, nx_state, ny, time]
        self.start_time_index = 0
        self.stop_time_index = reader[task].shape[-1]
     
        inputs = torch.tensor(reader[task][:nsamples, ::self.xStep, ::self.yStep, \
                                        self.start_time_index: self.start_time_index + (self.T_in*self.tStep): self.tStep], \
                                        dtype=torch.float)
        
        outputs = torch.tensor(reader[task][:nsamples, ::self.xStep, ::self.yStep, \
                                            self.start_time_index + (self.T_in*self.tStep): self.start_time_index + \
                                            (self.T_in + self.T)*self.tStep: self.tStep],\
                                            dtype=torch.float)
        print(f"input data for {task}:{inputs.shape}")
        print(f"output data for {task}: {outputs.shape}")
        assert (self.nx_state == outputs.shape[-3])
        assert (self.ny == outputs.shape[-2])
        assert (self.T ==outputs.shape[-1])
        
        if self.dim == 'FNO3D':
            # input_normalizer = UnitGaussianNormalizer(inputs)
            # inputs = input_normalizer.encode(inputs)
            # output_normalizer = UnitGaussianNormalizer(outputs)
            # outputs = output_normalizer.encode(outputs)
            
            inputs = inputs.reshape(nsamples, self.nx_state, self.ny, 1, self.T_in).repeat([1,1,1,self.T,1])
            print(f"Input data after reshaping for {task}:{inputs.shape}")
        
        print(f'Total {task} data: {inputs.shape[0]}')
        return inputs, outputs

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
        dryRun=False, verbose=False, **kwargs):
    assert inSize == 1, "inSize != 1 not implemented yet ..."
    simDirsSorted = sorted(glob.glob(f"{dataDir}/simu_*"), key=lambda f: int(f.split('simu_',1)[1]))
    nSimu = int(kwargs.get("nSimu", len(simDirsSorted)))
    simDirs = simDirsSorted[:nSimu]
    print('Using Simulations:')
    for s in simDirs:
        print(f" -- {s}")

    # -- retrieve informations from first simulation
    outFiles = OutputFiles(f"{simDirs[0]}/run_data")

    times = outFiles.times().ravel()
    dtData = times[1]-times[0]
    dtInput = dtData*outStep  # noqa: F841 (used lated by an eval call)
    dtSample = dtData*inStep  # noqa: F841 (used lated by an eval call)
    xGrid, yGrid = outFiles.x, outFiles.y  # noqa: F841 (used lated by an eval call)
    
    iBeg = int(kwargs.get("iBeg", 0))
    iEnd = int(kwargs.get("iEnd", sum(outFiles.nFields)))
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

    dataShape = (nSamples*nSimu, *outFiles.shape)
    print(f'data shape: {dataShape}')
    inputs = dataset.create_dataset("inputs", dataShape)
    outputs = dataset.create_dataset("outputs", dataShape)
    for iSim, dataDir in enumerate(simDirs):
        outFiles = OutputFiles(f"{dataDir}/run_data")
        print(f" -- sampling data from {outFiles.folder}")
        for iSample, iField in enumerate(sRange):
            if verbose:
                print(f"\t -- creating sample {iSample+1}/{nSamples}")
            inpt, outp = outFiles.fields(iField), outFiles.fields(iField+outStep).copy()
            if outType == "update":
                outp -= inpt
                if outScaling != 1:
                    outp *= outScaling
            inputs[iSim*nSamples + iSample] = inpt
            outputs[iSim*nSamples + iSample] = outp
    dataset.close()
    print(" -- done !")
    

def getDataLoaders(dataFile, trainRatio=0.8, batchSize=20, seed=None, use_distributed_sampler=False):
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
        generator = torch.Generator().manual_seed(seed)
        trainSet, valSet = random_split(
            dataset, [trainSize, valSize], generator=generator)
        
    # Reconfigure DataLoaders to use a DistributedSampler for
    # distributed data parallel mode
    if use_distributed_sampler:
        train_sampler = DistributedSampler(trainSet, num_replicas=get_world_size(), rank=get_local_rank(), shuffle=True)
        val_sampler = DistributedSampler(valSet, num_replicas=get_world_size(), rank=get_local_rank(), shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    trainLoader = DataLoader(trainSet, batch_size=batchSize, sampler=train_sampler, shuffle=(train_sampler is None), pin_memory=True, num_workers=4)
    valLoader = DataLoader(valSet, batch_size=batchSize, sampler=val_sampler, shuffle=False, pin_memory=True, num_workers=4)

    return trainLoader, valLoader, dataset
