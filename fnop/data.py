"""
Minimal utilities to generate training and validation data
"""
import h5py
import torch as th
from torch.utils.data import Dataset, DataLoader, random_split, Subset


class HDF5Dataset(Dataset):

    def __init__(self, dataFile):
        self.file = h5py.File(dataFile, 'r')
        self.inputs = self.file['inputs']
        self.outputs = self.file['outputs']
        assert len(self.inputs) == len(self.outputs), \
            f"different sample number for inputs and outputs ({len(self.inputs)},{len(self.outputs),})"

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx]
        output = self.outputs[idx]
        return th.tensor(input), th.tensor(output)

    def __del__(self):
        self.file.close()


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

    return trainLoader, valLoader
