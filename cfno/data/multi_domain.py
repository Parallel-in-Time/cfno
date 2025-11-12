
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from collections import defaultdict
from cfno.data.preprocessing import HDF5Dataset

def variable_tensor_collate_fn(batch):
    """
    Groups tensors of the same shape together and batches them separately.
    returns [nPatch_per_sample, trainSamples//nPatch_per_sample, 4, sx,sy]
    """
    grouped_tensors_inp = defaultdict(list)
    grouped_tensors_out = defaultdict(list)
    for element in batch:
        key = tuple(element[0].shape)               # input and output have same shape
        grouped_tensors_inp[key].append(element[0])
        grouped_tensors_out[key].append(element[0])
        
    batched_inp = [torch.stack(tensors) for tensors in grouped_tensors_inp.values()]  
    batched_out = [torch.stack(tensors) for tensors in grouped_tensors_out.values()]
    
    nBatched_lists = len(batched_inp)
    batchSize = [batched_inp[iBatch].shape[0] for iBatch in range(nBatched_lists)]

    # When using fixed domain sampling wtih add_fullGrid, make balanced batchSizes
    if len(set(batchSize)) > 1:
        min_batchSize = min(x for x in batchSize if x > 1)
        new_batched_inp  = []
        new_batched_out  = []
        for iBatch in range(nBatched_lists):
             split_inp = torch.split(batched_inp[iBatch], split_size_or_sections=min_batchSize, dim=0)
             split_out = torch.split(batched_out[iBatch], split_size_or_sections=min_batchSize, dim=0)
             new_batched_inp += split_inp
             new_batched_out += split_out
        # for i in range(len(new_batched_inp)):
        #     print(f'{i}: {new_batched_inp[i].shape}', flush=True)
        return (new_batched_inp, new_batched_out)

    # for i in range(len(batched_inp)):
    #     print(f'{i}: {batched_inp[i].shape}', flush=True)
    return (batched_inp, batched_out)  

class RandomDomainDataset(HDF5Dataset):
    """
        Creating dataset by dividing full grid (nX,nY) into nPatch_per_sample different
        random sized patches per epoch of (sX,sY).

        Args:
            dataFile (hdf5): data file 
            pad_to_fullGrid (bool, optional): Pads (sX,sY) into (nX,nY) grid with zeros.
            use_fixedPatch_startIdx (bool, optional): To divide full grid (nX,nY) into nPatch_per_sample 
                                                       (sX,sY) sized patches starting from same index
                                                       per epoch. Defaults to False.
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

        super().__init__(dataFile, **kwargs)
        self.nPatch_per_sample = nPatch_per_sample
        self.pad_to_fullGrid = pad_to_fullGrid
        self.use_fixedPatch_startIdx = use_fixedPatch_startIdx 
        self.use_minLimit = use_minLimit

        if not self.pad_to_fullGrid:
            self.use_minLimit = True
        
        self.slices = kwargs.get('slices', self.find_patchSize())
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
            yPatch_startIdx = self.patch_startIdx[iPatch][1]
        else:
            xPatch_startIdx = random.randint(0, self.nX - sX)
            yPatch_startIdx = random.randint(0, self.nY - sY)
        
        patch_padding[0] = 0 if xPatch_startIdx == 0 or (xPatch_startIdx - patch_padding[0]) < 0 else patch_padding[0]
        patch_padding[1] = 0 if (xPatch_startIdx + sX + patch_padding[1]) >= self.nX else patch_padding[1]
        patch_padding[2] = 0 if yPatch_startIdx == 0 or (yPatch_startIdx- patch_padding[2]) < 0 else patch_padding[2]
        patch_padding[3] = 0 if (yPatch_startIdx + sY + patch_padding[3]) >= self.nY else patch_padding[3]

        if self.pad_to_fullGrid:
            inpt, outp = np.zeros_like(inpt_grid), np.zeros_like(outp_grid)
            inpt[:, :(sX + patch_padding[0] + patch_padding[1]), 
                    :(sY + patch_padding[2] + patch_padding[3])] = inpt_grid[:, xPatch_startIdx - patch_padding[0]: (xPatch_startIdx + sX) + patch_padding[1], 
                                                                                yPatch_startIdx - patch_padding[2]: (yPatch_startIdx + sY) + patch_padding[3]]
            outp[:,:(sX + patch_padding[0] + patch_padding[1]),
                :(sY + patch_padding[2] + patch_padding[3])] = outp_grid[:, xPatch_startIdx - patch_padding[0]: (xPatch_startIdx + sX) + patch_padding[1], 
                                                                            yPatch_startIdx - patch_padding[2]: (yPatch_startIdx + sY) + patch_padding[3]]
        else:
            inpt = inpt_grid[:, xPatch_startIdx - patch_padding[0]: (xPatch_startIdx + sX) + patch_padding[1], yPatch_startIdx - patch_padding[2]: (yPatch_startIdx + sY) + patch_padding[3]]
            outp = outp_grid[:, xPatch_startIdx - patch_padding[0]: (xPatch_startIdx + sX) + patch_padding[1], yPatch_startIdx - patch_padding[2]: (yPatch_startIdx + sY) + patch_padding[3]]
        
        return torch.tensor(inpt), torch.tensor(outp)

    def find_patchSize(self):
        """
        List containing patch sizes
        """
        slices = []
        nX_min, nY_min = (self.calc_sliceMin(self.nX, self.kX), self.calc_sliceMin(self.nY, self.kY)) if self.use_minLimit else (1, 1)
        for _ in range(self.nPatch_per_sample):
            sX = random.randint(nX_min, self.nX)
            sY = random.randint(nY_min, self.nY)
            slices.append((sX, sY))
        return slices

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
            pad_to_fullGrid (bool, optional): Pad (sX,sY) into (nX,nY) grid with zeros.
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

        super().__init__(dataFile, **kwargs)
        self.nPatch_per_sample = nPatch_per_sample
        self.pad_to_fullGrid = pad_to_fullGrid
        self.use_fixedPatch_startIdx = use_fixedPatch_startIdx 
        self.use_minLimit = use_minLimit
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
            yPatch_startIdx = self.patch_startIdx[iPatch][1]
        elif self.use_orderedSampling:
            if sX == self.nX and sY == self.nY:
                xPatch_startIdx = 0
                yPatch_startIdx = 0
            else:
                xPatch_startIdx = (iPatch % (self.nX//sX)) * sX
                yPatch_startIdx = (iPatch % (self.nY//sY)) * sY            
        else:
            xPatch_startIdx = random.randint(0, self.nX - sX)
            yPatch_startIdx= random.randint(0, self.nY - sY)
        
        patch_padding[0] = 0 if xPatch_startIdx == 0 or (xPatch_startIdx - patch_padding[0]) < 0 else patch_padding[0]
        patch_padding[1] = 0 if (xPatch_startIdx + sX + patch_padding[1]) >= self.nX else patch_padding[1]
        patch_padding[2] = 0 if yPatch_startIdx == 0 or (yPatch_startIdx - patch_padding[2]) < 0 else patch_padding[2]
        patch_padding[3] = 0 if (yPatch_startIdx+ sY + patch_padding[3]) >= self.nY else patch_padding[3]

        if self.pad_to_fullGrid:
            inpt, outp = np.zeros_like(inpt_grid), np.zeros_like(outp_grid)
            inpt[:, :(sX + patch_padding[0] + patch_padding[1]), 
                    :(sY + patch_padding[2] + patch_padding[3])] = inpt_grid[:, xPatch_startIdx - patch_padding[0]: (xPatch_startIdx + sX) + patch_padding[1], 
                                                                                yPatch_startIdx - patch_padding[2]: (yPatch_startIdx + sY) + patch_padding[3]]
            outp[:,:(sX + patch_padding[0] + patch_padding[1]),
                :(sY + patch_padding[2] + patch_padding[3])] = outp_grid[:, xPatch_startIdx - patch_padding[0]: (xPatch_startIdx + sX) + patch_padding[1], 
                                                                            yPatch_startIdx - patch_padding[2]: (yPatch_startIdx + sY) + patch_padding[3]]
        else:
            inpt = inpt_grid[:, xPatch_startIdx - patch_padding[0]: (xPatch_startIdx + sX) + patch_padding[1], yPatch_startIdx - patch_padding[2]: (yPatch_startIdx + sY) + patch_padding[3]]
            outp = outp_grid[:, xPatch_startIdx - patch_padding[0]: (xPatch_startIdx + sX) + patch_padding[1], yPatch_startIdx - patch_padding[2]: (yPatch_startIdx + sY) + patch_padding[3]]
        
        return torch.tensor(inpt), torch.tensor(outp)

    def find_patchSize(self):
        """
        List containing patch sizes
        """
        slices = []
        nX_min, nY_min = (self.calc_sliceMin(self.nX, self.kX), self.calc_sliceMin(self.nY, self.kY)) if self.use_minLimit else (1, 1)
        self.valid_sX = [sx for sx in range(nX_min,self.nX) if self.nX % sx == 0]
        self.valid_sY = [sy for sy in range(nY_min,self.nY) if self.nY % sy == 0]
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
        print(f" -- patches (per sample): {self.slices}")
        print(f" -- padding (per patch): {self.padding}")
        if self.use_minLimit:
            print(f"Min nX & nY for patch computed using {self.kX, self.kY} modes")
        if self.use_fixedPatch_startIdx:
            print(f" -- patch start index (per epoch): {self.patch_startIdx}")

def get_multi_domain_dataLoaders(dataFile, trainRatio=0.8, batchSize=20, seed=None, 
                            use_domainSampling=False, 
                            use_fixedPatchSize=False,
                            pad_to_fullGrid=False,
                            use_orderedSampling=False,
                            use_fixedPatch_startIdx=False,
                            nPatch_per_sample=1,
                            use_minLimit=False,
                            padding=[0,0,0,0], 
                            add_fullGrid=False,
                   **kwargs):

    if use_domainSampling:
        if use_fixedPatchSize or use_orderedSampling:    
            dataset = FixedDomainDataset(dataFile,
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
        if add_fullGrid:
            dataset.slices.append((dataset.nX, dataset.nY))
            dataset.nPatch_per_sample = len(dataset.slices)
            if use_fixedPatch_startIdx and not use_orderedSampling:
                dataset.patch_startIdx.append((0,0))
    else:
        dataset = HDF5Dataset(dataFile)


    dataset.printInfos()
    if (use_domainSampling and not pad_to_fullGrid and not (use_fixedPatchSize or use_orderedSampling)) or \
        ((use_fixedPatchSize or use_orderedSampling) and (add_fullGrid or padding != [0,0,0,0])):
        nBatches = len(dataset)*dataset.nPatch_per_sample
        collate_fn = variable_tensor_collate_fn
    else:
        nBatches = len(dataset)
        collate_fn = None

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

    if (use_domainSampling and not pad_to_fullGrid and not (use_fixedPatchSize or use_orderedSampling)) or \
        ((use_fixedPatchSize or use_orderedSampling) and (add_fullGrid or padding != [0,0,0,0])):
        train_batchSize = len(trainSet)
        valid_batchSize = len(valSet)
    else:
        train_batchSize = batchSize
        valid_batchSize = batchSize
        
    trainLoader = DataLoader(trainSet, batch_size=train_batchSize, shuffle=True, num_workers=0, collate_fn=collate_fn, pin_memory=True)
    valLoader = DataLoader(valSet, batch_size=valid_batchSize, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=True)

    return trainLoader, valLoader, dataset
