"""
Minimal utilities to generate training and validation data
"""

import h5py
import torch
import glob
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from collections import defaultdict
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
 
class RandomDomainDataset(HDF5Dataset):
    """
        Creating dataset by dividing full grid (nX,nY) into nPatch_per_sample different
        random sized patches per epoch of (sX,sY).

        Args:
            dataFile (hdf5): data file 
            pad_to_fullGrid (bool, optional): Embeds (sX,sY) into (nX,nY) zero grid
            use_fixedPatch_startIdx (bool, optional): To divide full grid (nX,nY) into nPatch_per_sample 
                                                       (sX,sY) sized patches starting from same index
                                                       per epoch . Defaults to False.
            nPatch_per_sample (int, optional): Number of patches per sample. Defaults to 1.
            use_minLimit (bool, optional): Restrict (sX,sY) to be > (2*kX -1, 2*kY-1). Defaults to False.
            padding (list, optional): Columns and rows to decode inflow information
                                     in format[left, right, bottom, top]. Defaults to [0,0,0,0] 
            slices (list, optional): Sizes of patch [[sX,sY]]. Defaults to [].
            patch_startIdx (list, optional): Starting index of patch. Defaults to [[0,0]].
            kX (int, optional): Number of fourier modes in x-axis. Defaults to 12.
            kY (int, optional): Number of fourier modes in y-axis. Defaults to 12.
            
        """
    def __init__(self, dataFile, 
                 pad_to_fullGrid=False, 
                 use_fixedPatch_startIdx=True,
                 nPatch_per_sample=1,
                 use_minLimit=True,
                 padding=[0,0,0,0],
                 **kwargs):

        super().__init__(dataFile)
        self.nPatch_per_sample = nPatch_per_sample
        self.pad_to_fullGrid = pad_to_fullGrid
        self.use_fixedPatch_startIdx = use_fixedPatch_startIdx 
        self.use_minLimit = use_minLimit
        self.kX = kwargs.get('kX', 12)
        self.kY = kwargs.get('kY', 12)

        if not self.pad_to_fullGrid:
            self.use_minLimit = True
        
        slices = kwargs.get('slices', self.find_patchSize())
        patch_startIdx = kwargs.get('patch_startIdx', [[0,0]])
        if self.use_fixedPatch_startIdx:
            if len(patch_startIdx) == len(self.slices):
                self.patch_startIdx = patch_startIdx
            else:
                self.patch_startIdx = self.find_patch_startIdx()

        self.padding = padding  #[left, right, bottom, top]

        assert len(self.slices) == self.nPatch_per_sample, "Number of slices doesn't match patches per sample"

            
    def __getitem__(self, idx):
        patch_padding = self.padding.copy()
        iSample = idx // self.nPatch_per_sample
        iPatch = idx % self.nPatch_per_sample
        inpt_grid, outp_grid = self.sample(iSample)
        sX, sY = self.slices[iPatch]
        if self.use_fixedPatch_startIdx:
            xPatch_startIdx = self.patch_startIdx[iPatch][0]
            yPatch_startIdx= self.patch_startIdx[iPatch][1]
        else:
            xPatch_startIdx = random.randint(0, self.nX - sX)
            yPatch_startIdx= random.randint(0, self.nY - sY)
        
        patch_padding[0] = 0 if xPatch_startIdx == 0 or (xPatch_startIdx - patch_padding[0]) < 0 else patch_padding[0]
        patch_padding[1] = 0 if (xPatch_startIdx + sX + patch_padding[1]) >= self.nX else patch_padding[1]
        patch_padding[2] = 0 if yPatch_startIdx == 0 or (yPatch_startIdx- patch_padding[2]) < 0 else patch_padding[2]
        patch_padding[3] = 0 if (yPatch_startIdx+ sY + patch_padding[3]) >= self.nY else patch_padding[3]

        if self.pad_to_fullGrid:
            inpt, outp = np.zeros_like(inpt_grid), np.zeros_like(outp_grid)
            inpt[:, :(sX + patch_padding[0] + patch_padding[1]), 
                    :(sY + patch_padding[2] + patch_padding[3])] = inpt_grid[:, xPatch_startIdx - patch_padding[0]: (xPatch_startIdx+sX) + patch_padding[1], 
                                                                            yPatch_startIdx- patch_padding[2]: (yPatch_start+sY) + patch_padding[3]]
            outp[:,:(sX + patch_padding[0] + patch_padding[1]),
                :(sY + patch_padding[2] + patch_padding[3])] = outp_grid[:, xPatch_startIdx - patch_padding[0]: (xPatch_startIdx+sX) + patch_padding[1], 
                                                                            yPatch_startIdx- patch_padding[2]: (yPatch_start+sY) + patch_padding[3]]
        else:
            inpt = inpt_grid[:, xPatch_startIdx - patch_padding[0]: (xPatch_startIdx+sX) + patch_padding[1], yPatch_startIdx- patch_padding[2]: (yPatch_start+sY) + patch_padding[3]]
            outp = outp_grid[:, xPatch_startIdx - patch_padding[0]: (xPatch_startIdx+sX) + patch_padding[1], yPatch_startIdx- patch_padding[2]: (yPatch_start+sY) + patch_padding[3]]
        
        return torch.tensor(inpt), torch.tensor(outp)

    def find_patchSize(self):
        """
        List containing patch sizes
        """
        slices = []
        nX_min, nY_min = (self.calc_sliceMin(self.nX, self.kX), self.calc_sliceMin(self.nY, self.kY)) if self.use_minLimit else (0, 0)
        for _ in range(self.nPatch_per_sample):
            sX = random.randint(nX_min, self.nX)
            sY = random.randint(nY_min, self.nY)
            slices.append((sX, sY))
        return slices

    def calc_sliceMin(self, n, modes):
        """
        Finding min number of points to satisfy
        n/2 +1 >= fourier modes
        """
        slice_min = 2*(modes-1)
        if slice_min < n:
            return slice_min
        else:
            print("Insufficient number of points to slice")
            return 0

    def find_patch_startIdx(self):
        """
        List containing patch starting index
        """
        patch_start = []
        for i in range(len(self.slices)):
            xPatch_startIdx = random.randint(0, self.nX - self.slices[i][0])
            yPatch_startIdx = random.randint(0, self.nY - self.slices[i][1])
            patch_start.append((xPatch_startIdx, yPatch_startIdx))
        return patch_start

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
        print(f" -- pad_to_fullGrid: {self.pad_to_fullGrid}")
        print(f" -- nPatch (per sample): {self.nPatch_per_sample}")
        print(f" -- patches (per sample): {self.slices}")
        print(f" -- padding (per patch): {self.padding}")
        if self.use_minLimit:
            print(f"Min nX & nY for patch computed using {self.kX, self.kY} modes")
        if self.use_fixedPatch_startIdx:
            print(f" -- patch start index (per epoch): {self.patch_startIdx}")

class FixedDomainDataset(HDF5Dataset):
    """
        Creating dataset by dividing full grid (nX,nY) into nPatch_per_sample of 
        (sX,sY) patches with overlapping

        Args:
            dataFile (hdf5): data file 
            use_orderedSampling (bool, optional): To divide full grid (nX,nY) into (nX//sX)*(nY//sY) 
                                                   exactly divisible (sX,sY) size patches w/o overlapping.
                                                   Defaults to False.
                                                   pad_to_fullGrid (bool, optional): Embeds (sX,sY) into (nX,nY) zero grid
            use_fixedPatch_startIdx (bool, optional): To divide full grid (nX,nY) into nPatch_per_sample 
                                                       (sX,sY) sized patches starting from same index
                                                       per epoch . Defaults to False.
            nPatch_per_sample (int, optional): Number of patches per sample. Defaults to 1.
            use_minLimit (bool, optional): Restrict (sX,sY) to be > (2*kX -1, 2*kY-1). Defaults to False.
            padding (list, optional): Columns and rows to decode inflow information
                                     in format[left, right, bottom, top]. Defaults to [0,0,0,0] 
            slices (list, optional): Sizes of patch [[sX,sY]]. Defaults to [].
            patch_startIdx (list, optional): Starting index of patch. Defaults to [[0,0]].
            kX (int, optional): Number of fourier modes in x-axis. Defaults to 12.
            kY (int, optional): Number of fourier modes in y-axis. Defaults to 12.
            
        """
    def __init__(self, dataFile, 
                 use_orderedSampling=False,
                 pad_to_fullGrid=False, 
                 use_fixedPatch_startIdx=True,
                 nPatch_per_sample=1,
                 use_minLimit=True,
                 padding=[0,0,0,0],
                 **kwargs):

        super().__init__(dataFile)
        self.nPatch_per_sample = nPatch_per_sample
        self.pad_to_fullGrid = pad_to_fullGrid
        self.use_fixedPatch_startIdx = use_fixedPatch_startIdx 
        self.use_minLimit = use_minLimit
        self.kX = kwargs.get('kX', 12)
        self.kY = kwargs.get('kY', 12)
        self.use_orderedSampling = use_orderedSampling

        if not self.pad_to_fullGrid:
            self.use_minLimit = True
        
        slices = kwargs.get('slices', [])
        if len(slices) == 0:
            single_slice = self.find_patchSize()
        else:
            single_slice = slices

        assert len(single_slice) == 1, f"{len(single_slice)} patch size given for uniform domain sampling"
        
        if self.use_orderedSampling:
            self.nPatch_per_sample = (self.nX // single_slice[0][0]) * (self.nY // single_slice[0][1])
        self.slices = single_slice * self.nPatch_per_sample
        
        assert not (self.use_fixedPatch_startIdx and self.use_orderedSampling), \
            "use_fixedPatch_startIdx and use_orderedSampling cannot be True at the same time."
        
        if self.use_fixedPatch_startIdx:
            patch_startIdx = kwargs.get('patch_startIdx', [])
            if len(patch_startIdx) == len(self.slices):
                self.patch_startIdx = patch_startIdx
            else:
                self.patch_startIdx = self.find_patch_startIdx()  
    
        self.padding = padding  #[left, right, bottom, top]
            
    def __getitem__(self, idx):
        patch_padding = self.padding.copy()
        iSample = idx // self.nPatch_per_sample
        iPatch = idx % self.nPatch_per_sample
        inpt_grid, outp_grid = self.sample(iSample)
        sX, sY = self.slices[iPatch]

        if self.use_fixedPatch_startIdx:
            xPatch_startIdx = self.patch_startIdx[iPatch][0]
            yPatch_startIdx= self.patch_startIdx[iPatch][1]
        elif self.use_orderedSampling:
            xPatch_startIdx = (iPatch // (self.nX//sX)) * sX
            yPatch_startIdx = (iPatch % (self.nY//sY)) * sY            
        else:
            xPatch_startIdx = random.randint(0, self.nX - sX)
            yPatch_startIdx= random.randint(0, self.nY - sY)
        
        patch_padding[0] = 0 if xPatch_startIdx == 0 or (xPatch_startIdx - patch_padding[0]) < 0 else patch_padding[0]
        patch_padding[1] = 0 if (xPatch_startIdx + sX + patch_padding[1]) >= self.nX else patch_padding[1]
        patch_padding[2] = 0 if yPatch_startIdx== 0 or (yPatch_startIdx- patch_padding[2]) < 0 else patch_padding[2]
        patch_padding[3] = 0 if (yPatch_startIdx+ sY + patch_padding[3]) >= self.nY else patch_padding[3]

        if self.pad_to_fullGrid:
            inpt, outp = np.zeros_like(inpt_grid), np.zeros_like(outp_grid)
            inpt[:, :(sX + patch_padding[0] + patch_padding[1]), 
                    :(sY + patch_padding[2] + patch_padding[3])] = inpt_grid[:, xPatch_startIdx - patch_padding[0]: (xPatch_startIdx+sX) + patch_padding[1], 
                                                                            yPatch_startIdx- patch_padding[2]: (yPatch_start+sY) + patch_padding[3]]
            outp[:,:(sX + patch_padding[0] + patch_padding[1]),
                :(sY + patch_padding[2] + patch_padding[3])] = outp_grid[:, xPatch_startIdx - patch_padding[0]: (xPatch_startIdx+sX) + patch_padding[1], 
                                                                            yPatch_startIdx- patch_padding[2]: (yPatch_startIdx+sY) + patch_padding[3]]
        else:
            inpt = inpt_grid[:, xPatch_startIdx - patch_padding[0]: (xPatch_startIdx+sX) + patch_padding[1], yPatch_startIdx- patch_padding[2]: (yPatch_start+sY) + patch_padding[3]]
            outp = outp_grid[:, xPatch_startIdx - patch_padding[0]: (xPatch_startIdx+sX) + patch_padding[1], yPatch_startIdx- patch_padding[2]: (yPatch_start+sY) + patch_padding[3]]
        
        return torch.tensor(inpt), torch.tensor(outp)

    def find_patchSize(self):
        """
        List containing patch sizes
        """
        slices = []
        self.valid_sX = [sx for sx in range(1,self.nX) if self.nX % sx == 0]
        self.valid_sY = [sy for sy in range(1,self.nY) if self.nY % sy == 0]
        # select a (sX,sY) randomly 
        sX = int(random.choice(self.valid_sX))  
        sY = int(random.choice(self.valid_sY))
        slices.append((sX,sY))
        return slices

    def find_patch_startIdx(self):
        """
        List containing patch starting index
        """
        patch_start = []
        for i in range(len(self.slices)):
            xPatch_startIdx = random.randint(0, self.nX - self.slices[i][0])
            yPatch_startIdx= random.randint(0, self.nY - self.slices[i][1])
            patch_start.append((xPatch_startIdx, yPatch_startIdx))
        return patch_start

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
        print(f" -- use_orderedSampling: {self.use_orderedSampling}")
        print(f" -- pad_to_fullGrid: {self.pad_to_fullGrid}")
        print(f" -- nPatch (per sample): {self.nPatch_per_sample}")
        print(f" -- patches (per sample): {self.slices[0]}")
        print(f" -- padding (per patch): {self.padding}")
        if self.use_minLimit:
            print(f"Min nX & nY for patch computed using {self.kX, self.kY} modes")
        if self.use_fixedPatch_startIdx:
            print(f" -- patch start index (per epoch): {self.patch_startIdx}")


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

def getDataLoaders(dataFile, trainRatio=0.8, batchSize=20, seed=None, 
                   use_domainSampling=False, 
                   use_fixedPatchSize=False,
                   pad_to_fullGrid=False,
                   use_orderedSampling=False,
                   use_fixedPatch_startIdx=False,
                   nPatch_per_sample=1,
                   use_minLimit=False,
                   padding=[0,0,0,0], 
                   **kwargs):

    if not use_domainSampling:
        dataset = HDF5Dataset(dataFile)
    else:
        if use_fixedPatchSize:    
            dataset =  FixedDomainDataset(dataFile,
                                          use_orderedSampling,
                                          pad_to_fullGrid, 
                                          use_fixedPatch_startIdx,
                                          nPatch_per_sample,
                                          use_minLimit,
                                          padding, 
                                          **kwargs)
        else:
            dataset = RandomDomainDataset(dataFile,
                                          pad_to_fullGrid, 
                                          use_fixedPatch_startIdx,
                                          nPatch_per_sample,
                                          use_minLimit,
                                          padding, 
                                          **kwargs)

    dataset.printInfos()

    nBatches = len(dataset)
    collate_fn = None

    train_batchSize = batchSize
    valid_batchSize = batchSize
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
        
    if use_domainSampling and not pad_to_fullGrid:
        train_batchSize = len(trainSet)
        valid_batchSize = len(valSet)
        collate_fn = variable_tensor_collate_fn
       
    trainLoader = DataLoader(trainSet, batch_size=train_batchSize, shuffle=True, num_workers=0, collate_fn=collate_fn, pin_memory=True)
    valLoader = DataLoader(valSet, batch_size=valid_batchSize, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=True)

    return trainLoader, valLoader, dataset

def variable_tensor_collate_fn(batch):
    """
    Groups tensors of the same shape together and batches them separately.
    """
    grouped_tensors_inp = defaultdict(list)
    grouped_tensors_out = defaultdict(list)

    for element in batch:
        key = tuple(element[0].shape)               # input and output have same shape
        # [trainSamples//nPatch_per_sample,4,sx,sy]
        grouped_tensors_inp[key].append(element[0])
        grouped_tensors_out[key].append(element[0])
        
    # Stack tensors in each group to give [nPatch_per_sample, trainSamples//nPatch_per_sample, 4, sx,sy]
    batched_tensors_inp = [torch.stack(tensors) for tensors in grouped_tensors_inp.values()]  
    batched_tensors_out = [torch.stack(tensors) for tensors in grouped_tensors_out.values()]
 
    return (batched_tensors_inp, batched_tensors_out)  