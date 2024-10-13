import torch as th

from fnop.data import getDataLoaders
from fnop.models.cfno2d import CFNO2D


def lossFunction(out, ref):
    """ out,ref[nBatch,nVar,nX,nZ] """
    refNorms = th.linalg.vector_norm(ref, ord=2, dim=(-2, -1))
    diffNorms = th.linalg.vector_norm(out-ref, ord=2, dim=(-2, -1))
    return th.mean(diffNorms/refNorms)


class Trainer:

    def __init__(self, dataFile, batchSize=20):
        self.trainLoader, self.valLoader = getDataLoaders(dataFile, batchSize=batchSize)
        self.dataFile = dataFile
        self.batchSize = batchSize

        self.model = CFNO2D(da=4, dv=4, du=4, kX=12, kY=6, nLayers=4)
        self.optimizer = th.optim.Adam(
            self.model.parameters(),
            lr=0.00039,
            weight_decay=1e-5
            )

    def train(self):
        size = len(self.trainLoader.dataset)
        batchSize = self.batchSize
        model = self.model
        optimizer = self.optimizer

        model.train()
        for iBatch, (inputs, outputs) in enumerate(self.trainLoader):
            pred = model(inputs)
            loss = lossFunction(pred, outputs)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss, current = loss.item(), iBatch*batchSize + len(inputs)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


    def test(self):
        nBatches = len(self.valLoader)
        testLoss = 0
        model = self.model

        model.eval()
        with th.no_grad():
            for input, output in self.valLoader:
                pred = model(input)
                testLoss += lossFunction(pred, output).item()
        testLoss /= nBatches
        print(f"Test Error: \n Avg loss: {testLoss:>8f} \n")


    def runTraining(self, nEpochs):
        for n in range(nEpochs):
            print(f"Epoch {n+1}\n-------------------------------")
            self.train()
            self.test()
        print("Done!")
