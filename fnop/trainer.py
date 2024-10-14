import torch as th

from fnop.data import getDataLoaders
from fnop.models.cfno2d import CFNO2D


def lossFunction(out, ref):
    """ out,ref[nBatch,nVar,nX,nZ] """
    refNorms = th.linalg.vector_norm(ref, ord=2, dim=(-2, -1))
    diffNorms = th.linalg.vector_norm(out-ref, ord=2, dim=(-2, -1))
    return th.mean(diffNorms/refNorms)


class Trainer:

    def __init__(self, dataFile, batchSize=20, xStep=1, zStep=1):
        self.trainLoader, self.valLoader = getDataLoaders(dataFile, batchSize=batchSize)
        self.dataFile = dataFile
        self.batchSize = batchSize
        self.xStep = xStep
        self.zStep = zStep

        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

        self.model = CFNO2D(
            da=4, dv=4, du=4,
            kX=8, kY=8, nLayers=2,
            ).to(self.device)
        self.optimizer = th.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-5
            )
        self.opt = "adam"
        self.scheduler = th.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=100.0,
            gamma=0.98)

        self.tLoss = self.idLoss()

    def setLearningRate(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def switchOptimizer(self, opt="lbfgs"):
        if opt == "lbfgs":
            self.optimizer = th.optim.LBFGS(
                self.model.parameters(),
                line_search_fn="strong_wolfe")
        else:
            raise ValueError(f"cannot switch to {opt} optimizer")
        self.opt = opt

    def train(self):
        size = len(self.trainLoader.dataset)
        batchSize = self.batchSize
        model = self.model
        optimizer = self.optimizer
        # scheduler = self.scheduler

        model.train()
        for iBatch, data in enumerate(self.trainLoader):
            inputs = data[0][..., ::self.xStep, ::self.zStep].to(self.device)
            outputs = data[1][..., ::self.xStep, ::self.zStep].to(self.device)
            pred = model(inputs)
            loss = lossFunction(pred, outputs)

            loss.backward()
            if self.opt == "adam":
                optimizer.step()
            elif self.opt == "lbfgs":
                def closure():
                    optimizer.zero_grad()
                    pred = model(inputs)
                    loss = lossFunction(pred, outputs)
                    loss.backward()
                    return loss
                optimizer.step(closure)
            optimizer.zero_grad()

            loss, current = loss.item(), iBatch*batchSize + len(inputs)
            print(f"loss: {loss:>7f} (target: {self.tLoss:>7f}) [{current:>5d}/{size:>5d}]")


    def test(self):
        nBatches = len(self.valLoader)
        testLoss = 0
        model = self.model

        model.eval()
        with th.no_grad():
            for data in self.valLoader:
                inputs = data[0].to(self.device)
                outputs = data[1].to(self.device)
                pred = model(inputs)
                testLoss += lossFunction(pred, outputs).item()
        testLoss /= nBatches
        print(f"Test Error: \n Avg loss: {testLoss:>8f} (target: {self.tLoss:>7f})\n")


    def idLoss(self, dataset="valid"):
        if dataset == "valid":
            loader = self.valLoader
        elif dataset == "train":
            loader = self.trainLoader
        else:
            ValueError(f"cannot compute id loss on {loader} dataset")
        nBatches = len(loader)
        avgLoss = 0
        model = self.model

        model.eval()
        with th.no_grad():
            for inputs, outputs in loader:
                avgLoss += lossFunction(inputs, outputs).item()
        avgLoss /= nBatches
        return avgLoss


    def runTraining(self, nEpochs):
        for n in range(nEpochs):
            print(f"Epoch {n+1}\n-------------------------------")
            self.train()
            self.test()
        print("Done!")


    def loadCheckpoint(self, filePath):
        checkpoint = th.load(filePath, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    def saveCheckpoint(self, filePath):
        th.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, filePath)
