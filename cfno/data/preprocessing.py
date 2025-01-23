"""
Minimal utilities to generate training and validation data
"""

import h5py
import torch
import glob
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from cfno.simulation.post import OutputFiles

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

    def __init__(self, dataFile, 
                 use_domain_sampling=False, 
                 nPatch_per_sample=1, 
                 use_min_limit=False,
                 padding=[0,0,0,0],
                 kX= 12, kY= 12,
                 slices=[(16,16)],
                 xPatch_start=0,
                 yPatch_start=0):
        
        self.file = h5py.File(dataFile, 'r')
        self.inputs = self.file['inputs']
        self.outputs = self.file['outputs']
        self.use_domain_sampling = use_domain_sampling
        self.nPatch_per_sample = nPatch_per_sample
        self.use_min_limit = use_min_limit
        self.kX = kX
        self.kY = kY
        xGrid, yGrid = self.grid
        self.nX = xGrid.size
        self.nY = yGrid.size
        self.xPatch_start = xPatch_start
        self.yPatch_start = yPatch_start

        if len(slices) != nPatch_per_sample:
            self.slices = self.find_patch_size()
        else:
            self.slices = slices

        self.padding = padding  #[left, right, bottom, top]
            
        assert len(self.inputs) == len(self.outputs), \
            f"different sample number for inputs and outputs ({len(self.inputs)},{len(self.outputs)})"

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.use_domain_sampling:
            patch_padding = self.padding.copy()
            iSample = idx // self.nPatch_per_sample
            iPatch = idx % self.nPatch_per_sample
            inpt_grid, outp_grid = self.sample(iSample)  
            inpt, outp = np.zeros_like(inpt_grid), np.zeros_like(outp_grid)
    
            sX, sY = self.slices[iPatch]
            if len(self.slices) == 1:
                xPatch_start = self.xPatch_start.copy()
                yPatch_start = self.yPatch_start.copy()
            else:
                xPatch_start = random.randint(0, self.nX - sX)
                yPatch_start = random.randint(0, self.nY - sY)
           
            if xPatch_start == 0:
                patch_padding[0] = 0
            if xPatch_start == (self.nX-sX):
                patch_padding[1] = 0
            if yPatch_start == 0:
                patch_padding[2] = 0
            if yPatch_start == (self.nY-sY):
                patch_padding[3] = 0
            
            # print(f"Input size: {inpt.shape}")
            # print(f'For patch {iPatch} of sample {iSample}')
            # print(f'(sx,sy): {sX,sY}, (x_start,y_start): {xPatch_start,yPatch_start}')
            inpt[:, :(sX + patch_padding[0] + patch_padding[1]), 
                    :(sY + patch_padding[2] + patch_padding[3])] = inpt_grid[:, xPatch_start - patch_padding[0]: (xPatch_start+sX) + patch_padding[1], 
                                                                              yPatch_start - patch_padding[2]: (yPatch_start+sY) + patch_padding[3]]
            outp[:,:(sX + patch_padding[0] + patch_padding[1]),
                   :(sY + patch_padding[2] + patch_padding[3])] = outp_grid[:, xPatch_start - patch_padding[0]: (xPatch_start+sX) + patch_padding[1], 
                                                                             yPatch_start - patch_padding[2]: (yPatch_start+sY) + patch_padding[3]]
        else:
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
        
    def find_patch_size(self):
        """
        List containing random patch sizes
        """
        slices = []
        if self.use_min_limit:
            nX_min = self.calc_slice_min(self.nX, self.kX)
            nY_min = self.calc_slice_min(self.nY, self.kY)
        else:
            nX_min, nY_min = 0, 0
        for i in range(self.nPatch_per_sample):
            sX = random.randint(nX_min, self.nX)
            sY = random.randint(nY_min, self.nY)
            slices.append((sX,sY))
        return slices

    def calc_slice_min(self, n, modes):
        """
        Finding min number of points to satisfy
        n/2 +1 >= modes
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
        if self.use_domain_sampling:
            print(f"-- nPatch (per sample): {self.nPatch_per_sample}")
            print(f" --patches (per sample): {self.slices}")
            print(f" --padding (per patch): {self.padding}")
            if self.use_min_limit:
                print(f" Min nX & nY for patch computed using {self.kX, self.kY} modes")

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

def getDataLoaders(dataFile, trainRatio=0.8, batchSize=20,
                   seed=None, use_domain_sampling=False, 
                   nPatch_per_sample=1,use_min_limit=False,
                   padding=[0,0,0,0],kX=12, kY=12):
    
    dataset = HDF5Dataset(dataFile,use_domain_sampling, 
                          nPatch_per_sample,use_min_limit,
                          padding,kX,kY)

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

    trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True)
    valLoader = DataLoader(valSet, batch_size=batchSize, shuffle=False)

    return trainLoader, valLoader, dataset
