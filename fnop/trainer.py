import torch as th

from fnop.data import getDataLoaders
from fnop.models.cfno2d import CFNO2D


def lossFunction(out, ref):
    """ out,ref[nBatch,nVar,nX,nY] """
    refNorms = th.linalg.vector_norm(ref, ord=2, dim=(-2, -1))
    diffNorms = th.linalg.vector_norm(out-ref, ord=2, dim=(-2, -1))
    return th.mean(diffNorms/refNorms)


class Trainer:

    def __init__(self, trainConfig:dict, modelConfig:dict, optimConfig:dict):

        # Training setup
        self.xStep, self.yStep = trainConfig.pop("xStep", 1), trainConfig.pop("yStep", 1)
        self.trainLoader, self.valLoader, self.dataset = getDataLoaders(**trainConfig)
        self.losses = {
            "model": {
                "valid": -1,
                "train": -1,
                },
            "id": {
                "valid": self.idLoss(),
                "train": self.idLoss("train"),
                }
            }

        # Model setup
        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
        self.model = CFNO2D(**modelConfig).to(self.device)

        # Optim configuration
        self.optim = optimConfig.pop("name", "adam")
        self.setOptimizer(self.optim, **optimConfig)
        self.epochs = 0


    @property
    def outType(self):
        # [()] needed to extract hdf5 scalar
        # .decode to convert from bytes to string
        return self.dataset.infos["outType"][()].decode("utf-8")

    @property
    def outScaling(self):
        return float(self.dataset.infos["outScaling"][()])

    @property
    def batchSize(self):
        return self.trainLoader.batch_size


    def idLoss(self, dataset="valid"):
        if dataset == "valid":
            loader = self.valLoader
        elif dataset == "train":
            loader = self.trainLoader
        else:
            ValueError(f"cannot compute id loss on {loader} dataset")
        nBatches = len(loader)
        avgLoss = 0
        outType = self.outType

        with th.no_grad():
            for inputs, outputs in loader:
                if outType == "solution":
                    avgLoss += lossFunction(inputs, outputs).item()
                elif outType == "update":
                    avgLoss += lossFunction(0*inputs, outputs).item()
                else:
                    raise ValueError(f"outType = {outType}")
        avgLoss /= nBatches
        return avgLoss


    def setOptimizer(self, name="adam", **params):
        if name == "adam":
            self.optimizer = th.optim.Adam(self.model.parameters(), **params)
        elif name == "lbfgs":
            baseParams = {
                "line_search_fn": "strong_wolfe"
                }
            params = {**baseParams, **params}
            self.optimizer = th.optim.LBFGS(self.model.parameters(), **params)
        else:
            raise ValueError(f"optim {name} not implemented yet")


    def setOptimizerParam(self, **params):
        for g in self.optimizer.param_groups:
            for name, val in params.items():
                g[name] = val


    def switchOptimizer(self, optim="lbfgs"):
        """DEPRECATED ..."""
        if optim == "lbfgs":
            self.optimizer = th.optim.LBFGS(
                self.model.parameters(),
                line_search_fn="strong_wolfe")
        else:
            raise ValueError(f"cannot switch to {optim} optimizer")
        self.optim = optim


    def train(self):
        """Train the model for one epoch"""
        nBatches = len(self.trainLoader.dataset)
        batchSize = self.batchSize
        model = self.model
        optimizer = self.optimizer
        avgLoss = 0
        idLoss = self.losses['id']['train']

        model.train()
        for iBatch, data in enumerate(self.trainLoader):
            inputs = data[0][..., ::self.xStep, ::self.yStep].to(self.device)
            outputs = data[1][..., ::self.xStep, ::self.yStep].to(self.device)

            pred = model(inputs)
            loss = lossFunction(pred, outputs)
            loss.backward()

            if self.optim == "adam":
                optimizer.step()

            elif self.optim == "lbfgs":
                def closure():
                    optimizer.zero_grad()
                    pred = model(inputs)
                    loss = lossFunction(pred, outputs)
                    loss.backward()
                    return loss
                optimizer.step(closure)

            optimizer.zero_grad()

            loss, current = loss.item(), iBatch*batchSize + len(inputs)
            print(f"loss: {loss:>7f} (id: {idLoss:>7f}) [{current:>5d}/{nBatches:>5d}]")
            avgLoss += loss

        avgLoss /= nBatches/batchSize
        print(f"Training: \n Avg loss: {avgLoss:>8f} (id: {idLoss:>7f})\n")
        self.losses["model"]["train"] = avgLoss
        self.epochs += 1


    def valid(self):
        """Validate the model for one epoch"""
        nBatches = len(self.valLoader)
        model = self.model
        avgLoss = 0
        idLoss = self.losses['id']['valid']

        model.eval()
        with th.no_grad():
            for data in self.valLoader:
                inputs = data[0].to(self.device)
                outputs = data[1].to(self.device)
                pred = model(inputs)
                avgLoss += lossFunction(pred, outputs).item()

        avgLoss /= nBatches
        print(f"Validation: \n Avg loss: {avgLoss:>8f} (id: {idLoss:>7f})\n")
        self.losses["model"]["valid"] = avgLoss


    def runTraining(self, nEpochs):
        for _ in range(nEpochs):
            print(f"Epoch {self.epochs+1}\n-------------------------------")
            self.train()
            self.valid()
        print("Done!")


    def loadCheckpoint(self, filePath):
        checkpoint = th.load(filePath, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        optim = checkpoint['optim']
        if self.optim != optim:
            self.setOptimizer(optim)
            self.optim = optim
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epochs = checkpoint['epochs']
        self.losses['model'] = checkpoint['losses']


    def saveCheckpoint(self, filePath):
        th.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'optim': self.optim,
            'epochs': self.epochs,
            'losses': self.losses["model"],
            }, filePath)
