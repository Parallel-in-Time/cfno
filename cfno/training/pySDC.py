import os,sys
import time
from pathlib import Path
from collections import OrderedDict
import torch as th
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from cfno.data.preprocessing import getDataLoaders
from cfno.models.cfno2d import CFNO2D
from cfno.losses import LOSSES_CLASSES
from cfno.utils import print_rank0
from cfno.communication import Communicator

class FourierNeuralOp:
    """
    Base superclass for a Fourier Neural Operator, that can :

    - train itself with some given model settings provided some training data
    - evaluate itself on given inputs using a provided model checkpoint

    """
    TRAIN_DIR = None

    LOSSES_FILE = None      # allows to write losses simultaneously in separate text file,
                            # for easier comparison between different training.
    USE_TENSORBOARD = True
    LOG_PRINT = False
    
    PHYSICS_LOSSES_FILE = None # to track the individual losses that are combined into the phyics loss


    def __init__(self, data:dict=None, model:dict=None, optim:dict=None, lr_scheduler:dict=None, 
                 checkpoint=None, parallel_strategy:dict=None):

        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
        self.DDP_enabled = False
        self.rank = int(os.getenv('RANK', '0'))
        self.world_size = int(os.getenv('WORLD_SIZE', '1'))
    
        if parallel_strategy is not None:
            gpus_per_node = parallel_strategy.pop("gpus_per_node", 4)
            self.DDP_enabled = parallel_strategy.pop("ddp", True)
            if self.DDP_enabled: 
                self.communicator = Communicator(gpus_per_node, self.rank)
                self.world_size = self.communicator.world_size
                assert  self.world_size > 1, 'More than 1 GPU required for ditributed training'
                self.device = self.communicator.device
                self.rank = self.communicator.rank
                self.local_rank = self.communicator.local_rank
        else:
            self.DDP_enabled = False
         
        # Inference-only mode
        if data is None and optim is None:
            assert checkpoint is not None, "need a checkpoint in inference-only evaluation"
            if model is not None:
                self.modelConfig = model
            self.dataset = None
            self.load(checkpoint, modelOnly=True)
            return

        # Data setup
        assert "dataFile" in data, ""
        self.data_config = {**data}
        self.xStep, self.yStep = data.pop("xStep", 1), data.pop("yStep", 1)
        data.pop("outType", None), data.pop("outScaling", None)  # overwritten by dataset
        self.trainLoader, self.valLoader, self.dataset = getDataLoaders(**data, use_distributed_sampler=self.DDP_enabled)
        # sample : [batchSize, 4, nX, nY]
        self.outType = self.dataset.outType
        self.outScaling = self.dataset.outScaling
        if loss is None:    # Use default settings
            loss = {
                "name": "VectorNormLoss",
                "absolute": False,
            }
        assert "name" in loss, "loss section in config must contain the 'name' parameter"
        name = loss.pop("name")
        try:
            LossClass = LOSSES_CLASSES[name]
        except KeyError:
            raise NotImplementedError(f"{name} loss not implemented, available are {list(LOSSES_CLASSES.keys())}")
        if "grids" in loss:
            loss["grids"] = self.dataset.grid
        self.lossFunction = LossClass(**loss, device=self.device)
        self.lossConfig = loss
        self.losses = {
            "model": {
                "avg_valid": -1,
                "avg_train": -1,
                "valid": -1,
                "train": -1
                },
            "id": {
                "valid": self.idLoss(),
                "train": self.idLoss("train"),
                }
            }
        
        # For backward compatibility, remove any "absLoss" in optim section and raise a warning if any
        old = optim.pop("absLoss", None)
        if old is not None: 
            print_rank0("WARNING : absLoss should be now specified with 'absolute' in the loss section for VectorNormLoss")

        # Set model
        self.setupModel(model)

        # Eventually load checkpoint if provided
        if checkpoint is not None: self.load(checkpoint)
        # /!\ overwrite model setup if checkpoint is loaded /!\

        self.setupOptimizer(optim)
        self.setupLRScheduler(lr_scheduler)

        self.epochs = 0
        self.tCompEpoch = 0
        self.gradientNormEpoch = 0.0
        if self.USE_TENSORBOARD and self.rank == 0:
            self.writer = SummaryWriter(self.fullPath("tboard"))
        
        # Print settings summary
        self.printInfos()


    # -------------------------------------------------------------------------
    # Setup and utility methods
    # -------------------------------------------------------------------------
    def setupModel(self, model):
        assert model is not None, "model configuration is required"
        self.model = CFNO2D(**model, dataset=self.dataset).to(self.device)
        self.modelConfig = {**self.model.config}
        self.modelConfig.pop('dataset', None)
        if self.DDP_enabled:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        th.cuda.empty_cache()   # in case another model was setup before ...

    def setupOptimizer(self, optim=None):
        if optim is None: optim = {"name": "adam", "lr" : 0.0001, "weight_decay": 1.0e-5}
        name = optim.pop("name", "adam")
        if name == "adam":
            OptimClass = th.optim.Adam
        elif name == "adamw":
            OptimClass = th.optim.AdamW
        elif name == "lbfgs":
            optim = {"line_search_fn": "strong_wolfe", **optim}
            OptimClass = LBFGS
        else:
            raise ValueError(f"optim {name} not implemented yet")
        self.optimizer = OptimClass(self.model.parameters(), **optim)
        self.optimConfig = optim
        self.optim = name
        th.cuda.empty_cache()   # in case another optimizer was setup before ...

    def setupLRScheduler(self, lr_scheduler=None):
        if lr_scheduler is None:
            lr_scheduler = {"scheduler": "StepLR", "step_size": 100.0, "gamma": 0.98}
        self.scheduler_config = lr_scheduler
        scheduler = lr_scheduler.pop('scheduler')
        self.scheduler_name = scheduler
        if scheduler == "StepLR":
            self.lr_scheduler = th.optim.lr_scheduler.StepLR(self.optimizer, **lr_scheduler)
        elif scheduler == "CosAnnealingLR":
            self.lr_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **lr_scheduler)
        else:
            raise ValueError(f"LR scheduler {scheduler} not implemented yet")
        

    def setOptimizerParam(self, **params):
        """
        Allows to change the optimizer parameter(s) dynamically during training
        (for instance learning rate, etc ...)
        """
        for g in self.optimizer.param_groups:
            for name, val in params.items():
                g[name] = val

    def printInfos(self):
        print_rank0("-"*80)
        print_rank0("Model settings")
        print_rank0(f" -- class: {self.model}")
        for key, val in self.modelConfig.items():
            print_rank0(f" -- {key}: {val}")
        print_rank0(f"Loss settings")
        print_rank0(f" -- name : {self.lossFunction.__class__.__name__}")
        for key, val in self.lossConfig.items():
            print_rank0(f" -- {key}: {val}")
        print_rank0("Optim settings")
        print_rank0(f" -- name : {self.optim}")
        for key, val in self.optimConfig.items():
            print_rank0(f" -- {key}: {val}")
        print_rank0(f"Scheduler: {self.scheduler_name}")
        for key,val in self.scheduler_config.items():
            print_rank0(f" -- {key}: {val}")
        print_rank0("Data settings")
        for key, val in self.data_config.items():
            print_rank0(f" -- {key}: {val}")
        # TODO: add more details here ...
        print_rank0("-"*80)
        

    def idLoss(self, dataset="valid"):
        if dataset == "valid":
            loader = self.valLoader
        elif dataset == "train":
            loader = self.trainLoader
        else:
            ValueError(f"cannot compute id loss on {loader} dataset")
        nBatches = len(loader)
        data_iter = iter(loader)
        avgLoss = 0
        outType = self.outType

        with th.no_grad():
            for iBatch in range(nBatches):
                inputs, outputs = next(data_iter)
                if outType == "solution":
                    avgLoss += self.lossFunction(inputs, outputs, inputs).item()
                elif outType == "update":
                    avgLoss += self.lossFunction(0*inputs, outputs, inputs).item()
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

        nSamples = len(self.trainLoader.dataset)
        nBatches = len(self.trainLoader)
        batchSize = self.trainLoader.batch_size
        data_iter = iter(self.trainLoader)

        model = self.model
        optimizer = self.optimizer
        scheduler = self.lr_scheduler
        avgLoss = 0.0
        gradsEpoch = 0.0
        idLoss = self.losses['id']['train']

        model.train()
        for iBatch in range(nBatches):
            data = next(data_iter)
            inp = data[0][..., ::self.xStep, ::self.yStep].to(self.device)
            ref = data[1][..., ::self.xStep, ::self.yStep].to(self.device)

            pred = model(inp)
            loss = self.lossFunction(pred, ref, inp)
            optimizer.zero_grad()
            loss.backward()

            # -> lbfgs optimizer requires a closure function when calling
            #    optimizer step, hence the if condition depending on the type
            #    of optimizer.
            if self.optim in ["adam", "adamw"]:
                optimizer.step()

            elif self.optim == "lbfgs":
                def closure():
                    optimizer.zero_grad()
                    pred = model(inp)
                    loss = self.lossFunction(pred, ref, inp)
                    loss.backward()
                    return loss
                optimizer.step(closure)

            grads = [param.grad.detach().flatten() for param in self.model.parameters() if param.grad is not None]
            gradsNorm = th.cat(grads).norm()
            if self.USE_TENSORBOARD and self.rank == 0:
                self.writer.add_histogram("Gradients/GradientNormBatch", gradsNorm, iBatch)
            gradsEpoch += gradsNorm

            loss, current = loss.item(), iBatch*batchSize + len(inp)
            if self.LOG_PRINT:
                print_rank0(f" At [{current}/{nSamples:>5d}] loss: {loss:>7f} (id: {idLoss:>7f}) -- lr: {optimizer.param_groups[0]['lr']}")
            avgLoss += loss

        scheduler.step()
        avgLoss /= nBatches
        gradsEpoch /= nBatches

        if self.DDP_enabled:
            # Obtain the global average loss.
            ddp_loss = th.Tensor([avgLoss]).to(self.device).clone()
            self.communicator.allreduce(ddp_loss,op=dist.ReduceOp.AVG)
            train_loss = ddp_loss.item()
        else:
            train_loss = avgLoss
        print_rank0(f"Training: \n Avg loss: {train_loss:>8f} (id: {idLoss:>7f})\n")
        self.losses["model"]["train"] = train_loss

        getPhysicsLosses = getattr(self.lossFunction, 'getLossValues', None)  
        if getPhysicsLosses is not None:
            partial_losses = getPhysicsLosses()
            if self.PHYSICS_LOSSES_FILE:
                with open(self.fullPath(self.PHYSICS_LOSSES_FILE), "a") as f:
                    f.write("{epochs}\t{velocity:1.18f}\t{buoyancy:1.18f}\t{pressure:1.18f}\t{divergence:1.18f}\t{data:1.18f}\n".format(
                        epochs=self.epochs,
                        velocity=partial_losses[0]/nBatches, buoyancy=partial_losses[1]/nBatches,
                        pressure=partial_losses[2]/nBatches, divergence=partial_losses[3]/nBatches, data=partial_losses[4]/nBatches))
            self.lossFunction.resetLossValues()
        self.gradientNormEpoch = gradsEpoch
        
        

    def valid(self):
        """Validate the model for one epoch"""
        nBatches = len(self.valLoader)
        model = self.model
        avgLoss = 0
        idLoss = self.losses['id']['valid']
        data_iter = iter(self.valLoader)

        model.eval()
        with th.no_grad():
            for iBatch in range(nBatches):
                data = next(data_iter)
                inp = data[0][..., ::self.xStep, ::self.yStep].to(self.device)
                ref = data[1][..., ::self.xStep, ::self.yStep].to(self.device)
                pred = model(inp)
                avgLoss += self.lossFunction(pred, ref, inp).item()

        avgLoss /= nBatches
        
        if self.DDP_enabled:
            # Obtain the global average loss.
            ddp_loss = th.Tensor([avgLoss]).to(self.device).clone()
            self.communicator.allreduce(ddp_loss,op=dist.ReduceOp.AVG)
            val_loss = ddp_loss.item()
        else:
            val_loss = avgLoss
            
        print_rank0(f"Validation: \n Avg loss: {val_loss:>8f} (id: {idLoss:>7f})\n")
        self.losses["model"]["valid"] = val_loss

    def learn(self, nEpochs):
        for _ in range(nEpochs):
            print_rank0(f"Epoch {self.epochs+1}\n-------------------------------")
            tBeg = time.perf_counter()
            self.train()
            self.valid()
            tComp = time.perf_counter()-tBeg
            self.tCompEpoch = tComp

            tBeg = time.perf_counter()
            self.monitor()
            tMonit = time.perf_counter()-tBeg

            self.epochs += 1
            print_rank0(f" --- End of epoch {self.epochs} (tComp: {tComp:1.2e}s, tMonit: {tMonit:1.2e}s) ---\n")

        print_rank0("Done!")

    def monitor(self): 
        if self.USE_TENSORBOARD and self.rank == 0:
            writer = self.writer
            writer.add_scalars("Losses", {
                "train_avg" : self.losses["model"]["train"],
                "valid_avg" : self.losses["model"]["valid"],
                "train_id": self.losses["id"]["train"],
                "valid_id": self.losses["id"]["valid"]
                }, self.epochs)
            writer.add_scalar("Gradients/GradientNormEpoch", self.gradientNormEpoch, self.epochs)
            writer.flush()

        if self.LOSSES_FILE and self.rank == 0:
            with open(self.fullPath(self.LOSSES_FILE), "a") as f:
                f.write("{epochs}\t{train:1.18f}\t{valid:1.18f}\t{train_id:1.18f}\t{valid_id:1.18f}\t{gradNorm:1.18f}\t{tComp}\n".format(
                    epochs=self.epochs,
                    train_id=self.losses["id"]["train"], valid_id=self.losses["id"]["valid"],
                    gradNorm=self.gradientNormEpoch, tComp=self.tCompEpoch, **self.losses["model"]))

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
            # Learning status
            'epochs': self.epochs,
            'losses': self.losses["model"],
            }
        if not modelOnly:
            infos.update({
            # Optimizer config and state
            'optim': self.optim,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict()
            })
            
        if self.rank == 0:
            th.save(infos, fullPath)
     
    def load(self, filePath, modelOnly=False):
        fullPath = self.fullPath(filePath)
        if self.DDP_enabled:
            map_location = {f'cuda:0': f'{self.device}'}
        else:
            map_location = self.device
        checkpoint = th.load(fullPath, weights_only=True, map_location=map_location)
      
        # Load model state (eventually config before)
        if 'model' in checkpoint:
            if 'nonLinearity' in checkpoint['model']:
                # for backward compatibility ...
                checkpoint['model']["non_linearity"] = checkpoint['model'].pop("nonLinearity")
            if hasattr(self, "modelConfig") and self.modelConfig != checkpoint['model']:
                for key, value in self.modelConfig.items():
                    if key not in checkpoint['model']:
                        checkpoint['model'][key] = value
                    if key == 'get_subdomain_output' and value == True:
                        checkpoint['model'][key] = value
                print_rank0("WARNING : different model settings in config file,"
                      " overwriting with config from checkpoint ...")
            print_rank0(f"Model: {checkpoint['model']}")
            self.setupModel(checkpoint['model'])
            
        state_dict = checkpoint['model_state_dict']
        # creating new OrderedDict for model trained without DDP but used now with DDP 
        # or model trained using DPP but used now without DDP
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if self.DDP_enabled:
                if k[:7] == 'module.':
                    name = k
                else:
                    name = 'module.'+ k
            else:
                if k[:7] == 'module.':
                    name = k[7:]
                else:
                    name = k    
            if v.dtype == th.complex64:
                new_state_dict[name] = th.view_as_real(v)
            else:
                new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.outType = checkpoint["outType"]
        self.outScaling = checkpoint["outScaling"]
        # Learning status
        self.epochs = checkpoint['epochs']
        try:
            self.losses['model'] = checkpoint['losses']
        except AttributeError:
            self.losses = {"model": checkpoint['losses']}
        if not modelOnly:
            # Load optimizer state
            optim = checkpoint['optim']
            self.setupOptimizer({"name": optim})
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            try:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            except KeyError:
                print_rank0("Learning rate scheduler is restarted!")



    # -------------------------------------------------------------------------
    # Inference method
    # -------------------------------------------------------------------------
    def __call__(self, u0, nEval=1):
        model = self.model
        multi = len(u0.shape) == 4
        if not multi: u0 = u0[None, ...]

        inpt = th.tensor(u0, device=self.device, dtype=th.get_default_dtype())

        model.eval()
        with th.no_grad():
            for i in range(nEval):
                outp = model(inpt)
                if self.outType == "update":
                    outp /= self.outScaling

                    # Mapping input shape to ouput to perform addition
                    if outp.shape == inpt.shape:
                        outp += inpt
                    else:
                        sliced_inpt = inpt[:,:,
                                      self.modelConfig['iXBeg']: self.modelConfig['iXEnd'],
                                      self.modelConfig['iYBeg']: self.modelConfig['iYEnd']]
                        # print(f'Sliced Input: {sliced_inpt.shape}')
                        outp += sliced_inpt
                inpt = outp

        if not multi:
            outp = outp[0]

        u1 = outp.cpu().detach().numpy()
        return u1
