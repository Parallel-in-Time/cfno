import os
import time
import math
from pathlib import Path
from collections import OrderedDict
import torch as th
import pickle
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from cfno.data.preprocessing import getDataLoaders
from cfno.models.cfno2d import CFNO2D
from cfno.models.fno import FNO
from cfno.losses import LOSSES_CLASSES
from cfno.utils import print_rank0, augment_batch_with_noise
from cfno.communication import Communicator
from .lbfgs import LBFGS

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


    def __init__(self, data:dict=None, model:dict=None, optim:dict=None, lr_scheduler:dict=None, loss:dict=None,
                 checkpoint=None, parallel_strategy:dict=None, model_class='CFNO', ndim=2, data_aug=False):

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
            self.model_class = model_class
            self.ndim = ndim
            self.load(checkpoint, modelOnly=True)
            return

        # Data setup
        assert "dataFile" in data, "Missing dataFile in data config"
        self.data_config = data.copy()
        self.xStep, self.yStep = self.data_config.pop("xStep", 1), self.data_config.pop("yStep", 1)
        self.data_config.pop("outType", 'solution')
        self.data_config.pop("outScaling", 1.0)
        self.trainLoader, self.valLoader, self.dataset = getDataLoaders(**self.data_config, use_distributed_sampler=self.DDP_enabled)
        # sample : [batchSize, 4, nX, nY] or [batchSize, 4, T, nX, nY]
        self.outType = self.dataset.outType
        self.outScaling = self.dataset.outScaling
        self.data_aug = data_aug
        if self.data_aug:
            print_rank0(f'Using data noise injection per batch')
        
        if loss is None:    # Use default settings
            loss = {
                "name": "VectorNormLoss",
                "absolute": False,
                "dim": ndim
            }
        assert "name" in loss, "loss section in config must contain the 'name' parameter"
        self.lossConfig = loss.copy()
        loss_class = LOSSES_CLASSES.get(self.lossConfig.pop("name"))
        if loss_class is None:
            raise NotImplementedError(f"Unknown loss type, available are {list(LOSSES_CLASSES.keys())}")
        if "grids" in loss:
            loss["grids"] = self.dataset.grid
        self.lossFunction = loss_class(**self.lossConfig, device=self.device)
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
        self.model_class = model_class
        self.ndim = ndim
 
        if checkpoint is not None: 
            self.load(checkpoint)
        else:
            self.setupModel(model)
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
        if self.model_class == 'CFNO':
            self.model = CFNO2D(**model).to(self.device)
        else:
            self.model = FNO(**model).to(self.device)
        self.modelConfig = model.copy()
        self.model.print_size()
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

    def setupLRScheduler(self, lr_scheduler=None):
        if lr_scheduler is None:
           lr_scheduler = {'scheduler': 'ConstantLR', 'total_iters': 1, 'factor': 1.0}
        self.scheduler_config = lr_scheduler
        scheduler = lr_scheduler.pop('scheduler')
        self.scheduler_name = scheduler
        if scheduler == "StepLR":
            self.lr_scheduler = th.optim.lr_scheduler.StepLR(self.optimizer, **lr_scheduler)
        elif scheduler == "CosAnnealingLR":
            self.lr_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **lr_scheduler)
        elif scheduler == "CosAnnealingWarmRestarts":
            self.lr_scheduler = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                            self.optimizer, **lr_scheduler
                        )
        elif scheduler == "ConstantLR":
            self.lr_scheduler = th.optim.lr_scheduler.ConstantLR(self.optimizer, **lr_scheduler)
        else:
            raise ValueError(f"LR scheduler {scheduler} not implemented yet")
        self.scheduler_config = lr_scheduler
        
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

        if self.data_aug:
           r = self.epochs/500
           eps_uv0, eps_uvf = 0.2, 0.002
           eps_b0,  eps_bf  = 0.05, 0.002
           eps_p0,  eps_pf  = 0.05, 0.002

           eps_uv = eps_uvf + (eps_uv0 - eps_uvf) * math.exp(-4 * r)
           eps_b  = eps_bf  + (eps_b0  - eps_bf)  * math.exp(-4 * r)
           eps_p  = eps_pf  + (eps_p0  - eps_pf)  * math.exp(-4 * r)

           noise_levels = (eps_uv, eps_uv, eps_b, eps_p)

        model.train()
        for iBatch in range(nBatches):
            data = next(data_iter)
            inp = data[0][..., ::self.xStep, ::self.yStep].to(self.device)
            ref = data[1][..., ::self.xStep, ::self.yStep].to(self.device)

            if self.data_aug:
                if inp.shape[1] == 4:
                    inp, ref = augment_batch_with_noise(inp, ref, noise_levels=noise_levels)
                    # inp, ref has shape (2B, 4, nx, ny)
                else:
                    inp, ref = augment_batch_with_noise(inp, ref, noise_levels=noise_levels[:3])
                    # inp, ref has shape (2B, 3, nx, ny)
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
            if len(grads) > 0:
                gradsNorm = th.cat(grads).norm()

                if self.USE_TENSORBOARD and self.rank == 0:
                    if th.isfinite(gradsNorm):
                        self.writer.add_scalar(
                            "Gradients/GradientNormBatch",
                            gradsNorm.item(),
                            iBatch
                        )

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
                if self.epochs == 0:
                    f.write("Epochs\t\tTrainLoss\t\tValidLoss\t\tTrainIdLoss\t\tValidIdLoss\t\tGradNorm\t\tComputeTime\n")
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
            'lr_scheduler': self.scheduler_name,
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict()
            })
            
        def is_picklable(obj):
            try:
                pickle.dumps(obj)
                return True
            except (pickle.PicklingError, TypeError, AttributeError):
                return False

        safe_infos = {k: v for k, v in infos.items() if is_picklable(v)}

        if self.rank == 0:
            th.save(safe_infos, fullPath)
     
    def load(self, filePath, modelOnly=False):
        fullPath = self.fullPath(filePath)
        if self.DDP_enabled:
            map_location = {f'cuda:0': f'{self.device}'}
        else:
            map_location = self.device
        
        weights_only = True if self.model_class == 'CFNO' else False
        checkpoint = th.load(fullPath, weights_only=weights_only, map_location=map_location)
      
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
            
        state_dict = checkpoint['model_state_dict']
        # creating new OrderedDict for model trained without DDP but used now with DDP 
        # or model trained using DPP but used now without DDP
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if self.DDP_enabled:
                if k == '_metadata':
                    name = k
                else:
                    if k[:7] == 'module.':
                        name = k
                    else:
                        name = 'module.'+ k
            else:
                if k[:7] == 'module.':
                    name = k[7:]
            # print(f'name: {name}, k: {k}')
            if isinstance(v, th.Tensor) and v.is_complex():
                v = th.view_as_real(v)
            if name.endswith(".weight.tensor"):
                name = name.replace(".weight.tensor", ".weight")
            if name not in new_state_dict:
                new_state_dict[name] = v
        if "_metadata" in new_state_dict:
            del new_state_dict["_metadata"]
        self.setupModel(checkpoint['model'])  
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
            resumed_lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
            # for g in self.optimizer.param_groups:
            #     g['lr'] = 3e-4

            # Move optimizer state tensors to correct device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, th.Tensor):
                        state[k] = v.to(self.device)
            if 'lr_scheduler' not in checkpoint.keys():
                self.scheduler_name = "CosAnnealingLR" 
                # "CosAnnealingWarmRestarts"
                # CosAnnealingLR
                self.scheduler_config = {"scheduler": self.scheduler_name}
                # self.lr_scheduler = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                #         self.optimizer,
                #         T_0=500,
                #         T_mult=1,
                #         eta_min=1e-6
                #     )
                self.lr_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
                self.lr_scheduler.last_epoch = self.epochs
            else:
                # self.lr_scheduler = None
                # self.scheduler_name = None
                lr_scheduler = {"scheduler": checkpoint['lr_scheduler']}
                lr_scheduler.update({'T_max': checkpoint['lr_scheduler_state_dict']['T_max']})
                # lr_scheduler.update({'T_0': checkpoint['lr_scheduler_state_dict']['T_0'],
                #                      'T_mult': checkpoint['lr_scheduler_state_dict']['T_mult'],
                #                      'eta_min': checkpoint['lr_scheduler_state_dict']['eta_min'],
                #                      'last_epoch': checkpoint['lr_scheduler_state_dict']['last_epoch']})
                self.setupLRScheduler(lr_scheduler)
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            print_rank0(f'LR scheduler sate: {self.lr_scheduler.state_dict()}')
            print_rank0(f'Resuming from checkpoint at epoch {self.epochs} with lr={resumed_lr}\
                          with scheduler {self.scheduler_name}....')
        # waiting for all ranks to load checkpoint
        if self.DDP_enabled:
            dist.barrier()

    # -------------------------------------------------------------------------
    # Inference method
    # -------------------------------------------------------------------------
    def __call__(self, u0, nEval=1):
        model = self.model
        multi = len(u0.shape) >= 4
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
