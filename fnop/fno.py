import os
import time
from pathlib import Path

import torch as th
from torch.utils.tensorboard import SummaryWriter

from fnop.data import getDataLoaders
from fnop.models.cfno2d import CFNO2D


def lossFunction(out, ref):
    """ out,ref[nBatch,nVar,nX,nY] """
    refNorms = th.linalg.vector_norm(ref, ord=2, dim=(-2, -1))
    diffNorms = th.linalg.vector_norm(out-ref, ord=2, dim=(-2, -1))
    return th.mean(diffNorms/refNorms)


class FourierNeuralOp:
    """
    Base superclass for a Fourier Neural Operator, that can :

    - train itself with some given model settings provided some training data
    - evaluate itself on given inputs using a provided model checkpoint

    """
    TRAIN_DIR = None
    def __init__(self, data:dict=None, model:dict=None, optim:dict=None, checkpoint=None):

        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

        # Inference-only mode
        if data is None and model is None and optim is None:
            assert checkpoint is not None, "need a checkpoint in inference-only evaluation"
            self.load(checkpoint, modelOnly=True)
            return

        # Data setup
        assert "dataFile" in data, ""
        self.xStep, self.yStep = data.pop("xStep", 1), data.pop("yStep", 1)
        data.pop("outType", None), data.pop("outScaling", None)  # overwritten by dataset
        self.trainLoader, self.valLoader, self.dataset = getDataLoaders(**data)
        # sample : [batchSize, 4, nX, nY]
        self.outType = self.dataset.outType
        self.outScaling = self.dataset.outScaling
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

        # Set model
        self.setupModel(model)

        # Eventually load checkpoint if provided
        if checkpoint is not None: self.load(checkpoint)
        # /!\ overwrite model setup if checkpoint is loaded /!\

        self.setupOptimizer(optim)
        self.epochs = 0
        self.writer = SummaryWriter(self.fullPath("tboard"))


    # -------------------------------------------------------------------------
    # Setup and utility methods
    # -------------------------------------------------------------------------
    def setupModel(self, model):
        assert model is not None, "model configuration is required"
        self.model = CFNO2D(**model).to(self.device)
        self.modelConfig = {**self.model.config}
        th.cuda.empty_cache()   # in case another model was setup before ...


    def setupOptimizer(self, optim=None):
        if optim is None: optim = {"name": "adam"}
        name = optim.pop("name", "adam")
        if name == "adam":
            self.optimizer = th.optim.Adam(self.model.parameters(), **optim)
        elif name == "lbfgs":
            baseParams = {
                "line_search_fn": "strong_wolfe"
                }
            params = {**baseParams, **optim}
            self.optimizer = th.optim.LBFGS(self.model.parameters(), **params)
        else:
            raise ValueError(f"optim {name} not implemented yet")
        self.optim = name
        th.cuda.empty_cache()   # in case another optimizer was setup before ...


    def setOptimizerParam(self, **params):
        for g in self.optimizer.param_groups:
            for name, val in params.items():
                g[name] = val


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


    @classmethod
    def fullPath(cls, filePath):
        if cls.TRAIN_DIR is not None:
            os.makedirs(cls.TRAIN_DIR, exist_ok=True)
            filePath = str(Path(cls.TRAIN_DIR) / filePath)
        return filePath


    # -------------------------------------------------------------------------
    # Training methods
    # -------------------------------------------------------------------------
    def train(self):
        """Train the model for one epoch"""
        nBatches = len(self.trainLoader.dataset)
        batchSize = self.trainLoader.batch_size
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


    def learn(self, nEpochs):
        for _ in range(nEpochs):
            print(f"Epoch {self.epochs+1}\n-------------------------------")
            tBeg = time.perf_counter()
            self.train()
            self.valid()
            self.monitor()
            tComp = time.perf_counter()-tBeg
            print(f" --- End of epoch {self.epochs} (tComp: {tComp:1.2e}s) ---\n")
        print("Done!")


    def monitor(self):
        writer = self.writer
        writer.add_scalars("Loss_avg", {
            "train" : self.losses["model"]["train"],
            "valid" : self.losses["model"]["valid"],
            "train_id": self.losses["id"]["train"],
            "valid_id": self.losses["id"]["valid"]
            }, self.epochs)
        writer.flush()


    def __del__(self):
        try:
            self.writer.close()
        except: pass


    # -------------------------------------------------------------------------
    # Save/Load methods
    # -------------------------------------------------------------------------
    def save(self, filePath, modelOnly=False):
        fullPath = self.fullPath(filePath)
        infos = {
            # Model config and state
            'model': self.modelConfig,
            'model_state_dict': self.model.state_dict(),
            'outType': self.outType,
            'outScaling': self.outScaling,
            }
        if not modelOnly:
            infos.update({
            # Optimizer config and state
            'optim': self.optim,
            'optimizer_state_dict': self.optimizer.state_dict(),
            # Learning status
            'epochs': self.epochs,
            'losses': self.losses["model"],
            })
        th.save(infos, fullPath)


    def load(self, filePath, modelOnly=False):
        fullPath = self.fullPath(filePath)
        checkpoint = th.load(fullPath, weights_only=True)
        # Load model state (eventually config before)
        if 'model' in checkpoint:
            if hasattr(self, "modelConfig") and self.modelConfig != checkpoint['model']:
                print("WARNING : different model settings in config file,"
                      " overwriting with config from checkpoint ...")
            self.setupModel(checkpoint['model'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.outType = checkpoint["outType"]
        self.outScaling = checkpoint["outScaling"]
        if not modelOnly:
            # Load optimizer state
            optim = checkpoint['optim']
            self.setupOptimizer({"name": optim})
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Learning status
            self.epochs = checkpoint['epochs']
            self.losses['model'] = checkpoint['losses']


    # -------------------------------------------------------------------------
    # Inference method
    # -------------------------------------------------------------------------
    def __call__(self, u0):
        model = self.model
        multi = len(u0.shape) == 4
        if not multi: u0 = u0[None, ...]

        inpt = th.tensor(u0, device=self.device)

        model.eval()
        with th.no_grad():
            outp = model(inpt)
            if self.outType == "update":
                outp /= self.outScaling
                outp += inpt
        if not multi:
            outp = outp[0]

        u1 = outp.cpu().detach().numpy()
        return u1
