import torch
import torch.nn as nn
import time
import sys
from tqdm import tqdm
from timeit import default_timer
from torch.utils.tensorboard import SummaryWriter
from fnop.utils import get_signal_handler, CudaMemoryDebugger, UnitGaussianNormalizer


_TRAIN_START_TIME = time.time()

class Trainer:
    """
    Trainer class to train fourier neural operators
    """
    def __init__(self,
                 model:nn.Module,
                 dim: str,
                 epochs:int,
                 dt:float,
                 nx: int,
                 ny: int,
                 T_in: int,
                 T:int,
                 xStep:int=1,
                 yStep:int=1,
                 tStep:int=1,
                 exit_signal_handler:bool=False,
                 exit_duration_in_mins=None,
                 device:str='cpu',
    ): 
        """

        Args:
            model (nn.Module): FNO model
            dim (str): 'FNO2D' or 'FNO3D'
            epochs (int): training epochs
            dt (float): delta timestep
            nx (int): size of x
            ny (int): size of y
            T_in (int): number of input timesteps
            T (int): number of output timesteps
            xStep (int, optional): slicing for x. Defaults to 1.
            yStep (int, optional): slicing for y. Defaults to 1.
            tStep (int, optional): time slicing. Defaults to 1.
            exit_signal_handler (bool, optional): dynamically save the checkpoint and shutdown the
                                                  training if SIGTERM is received
            exit_duration_in_mins (int, optional): exit the program after this many minutes.
            device (str, optional): cpu or cuda
        """
        super().__init__()
        self.model = model
        self.dim = dim
        self.epochs = epochs
        self.dt = dt
        self.nx = nx
        self.ny = ny
        self.T_in = T_in
        self.T = T
        self.xStep = xStep
        self.yStep = yStep
        self.tStep = tStep
        self.exit_signal_handler = exit_signal_handler
        self.exit_duration_in_mins = exit_duration_in_mins
        self.device = device
        self.memory = CudaMemoryDebugger(print_mem=True)
        self.nx_state = 4*self.nx
    
    def fno2d_train_single_epoch (self, tepoch, train_loader,
                                  nTrain, training_loss,
    ):
        """
        Perform one epoch training for FNO2D recurrent in time 

        Args:
            train_loader (torch.utils.data.DataLoader): training dataloaders
            nTrain (int): number of training samples
            training_loss  : training loss
            
        Returns:
            train_error (float): training error for one epoch
            train_step_error (float): training recurrence step error for one epoch
            avg_loss (float): average train error per individual sample
            grads_norm_epoch (float): torch gradient norm for one epoch
        """
        
        self.model.train()
        # memory.print("After model2d.train()")
        
        train_l2_step = 0.0
        train_l2_full = 0.0
        grads_norm_epoch = 0.0
        avg_loss = 0.0
        
        for step, (xx, yy) in enumerate(train_loader):
            loss = 0
            xx = xx.to(self.device)
            yy = yy.to(self.device)
            # self.memory.print("After loading first batch")

            for t in tqdm(range(0, self.T, self.tStep), desc="Train loop"):
                y = yy[..., t:t + self.tStep]
                im = self.model(xx)
                loss += training_loss(im.reshape(self.batch_size, -1), y.reshape(self.batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)
                    
                # print(f"{t}: y={y.shape},x={xx.shape},pred={pred.shape}")
                xx = torch.cat((xx[..., self.tStep:], im), dim=-1)
                # print(f"{t}: new_xx={xx.shape}")

            train_l2_step += loss.item()
            l2_full = training_loss(pred.reshape(self.batch_size, -1), yy.reshape(self.batch_size, -1))
            train_l2_full += l2_full.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            grads = [param.grad.detach().flatten() for param in self.model.parameters() if param.grad is not None]
            grads_norm = torch.cat(grads).norm()
            self.tensorboard_writer.add_histogram("train/GradNormStep", grads_norm, step)
            grads_norm_epoch += grads_norm
            
            # self.memory.print("After backwardpass")
            
            tepoch.set_postfix({'Batch': step + 1, 'Train l2 loss (in progress)': train_l2_full,\
                    'Train l2 step loss (in progress)': train_l2_step})
        

        train_error = train_l2_full / len(train_loader)
        train_step_error = train_l2_step /len(train_loader) / (self.T / self.tStep)
        grads_norm_epoch = grads_norm_epoch / nTrain
        avg_loss = train_l2_full/ nTrain
        
        return train_error, train_step_error, avg_loss, grads_norm_epoch
    
    def cfno2d_train_single_epoch (self, tepoch, train_loader,
                                  nTrain, training_loss,
    ):
        """
        Perform one epoch training for CFNO2D recurrent in time 

        Args:
            train_loader (torch.utils.data.DataLoader): training dataloaders
            nTrain (int): number of training samples
            training_loss  : training loss
            
        Returns:
            train_error (float): training error for one epoch
            train_step_error (float): training recurrence step error for one epoch
            avg_loss (float): average train error per individual sample
            grads_norm_epoch (float): torch gradient norm for one epoch
        """
        
        self.model.train()
        # memory.print("After model2d.train()")
        
        train_l2_step = 0.0
        train_l2_full = 0.0
        grads_norm_epoch = 0.0
        avg_loss = 0.0
        
        for step, (xx, yy) in enumerate(train_loader):
            loss = 0
            xx = xx.to(self.device)
            yy = yy.to(self.device)
            # self.memory.print("After loading first batch")

            for t in tqdm(range(0, self.T, self.tStep), desc="Train loop"):
                y = yy[..., t:t + self.tStep]
                im = self.model(xx)
                loss += training_loss(im.reshape(self.batch_size, -1), y.reshape(self.batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)
                    
                # print(f"{t}: y={y.shape},x={xx.shape},pred={pred.shape}")
                xx = torch.cat((xx[..., self.tStep:], im), dim=-1)
                # print(f"{t}: new_xx={xx.shape}")

            train_l2_step += loss.item()
            l2_full = training_loss(pred.reshape(self.batch_size, -1), yy.reshape(self.batch_size, -1))
            train_l2_full += l2_full.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            grads = [param.grad.detach().flatten() for param in self.model.parameters() if param.grad is not None]
            grads_norm = torch.cat(grads).norm()
            self.tensorboard_writer.add_histogram("train/GradNormStep", grads_norm, step)
            grads_norm_epoch += grads_norm
            
            # self.memory.print("After backwardpass")
            
            tepoch.set_postfix({'Batch': step + 1, 'Train l2 loss (in progress)': train_l2_full,\
                    'Train l2 step loss (in progress)': train_l2_step})
        

        train_error = train_l2_full / len(train_loader)
        train_step_error = train_l2_step /len(train_loader) / (self.T / self.tStep)
        grads_norm_epoch = grads_norm_epoch / nTrain
        avg_loss = train_l2_full/ nTrain
        
        return train_error, train_step_error, avg_loss, grads_norm_epoch
    
    def fno3d_train_single_epoch (self, tepoch, train_loader,
                                  nTrain, training_loss,
    ):
        """
        Perform one epoch training for FNO3D

        Args:
            train_loader (torch.utils.data.DataLoader): training dataloaders
            nTrain (int): number of training samples
            training_loss  : training loss
            
        Returns:
            train_error (float): training error for one epoch
            train_mse (float): training mean squared error for one epoch
            avg_loss (float): average train error per individual sample
            grads_norm_epoch (float): torch gradient norm for one epoch
        """
        
        self.model.train()
        # memory.print("After model2d.train()")
        # torch.autograd.set_detect_anomaly(True) 
        train_mse_local = 0
        train_l2 = 0
        grads_norm_epoch = 0.0
        avg_loss = 0.0
        
        for step, (xx, yy) in enumerate(train_loader):
            xx = xx.to(self.device)
            yy = yy.to(self.device)
            # self.memory.print("After loading first batch")

            pred = self.model(xx).view(self.batch_size, self.nx_state, self.ny, self.T)

            # yy = UnitGaussianNormalizer(yy).decode(yy)
            # pred = UnitGaussianNormalizer(pred).decode(pred)
            
            l2 = training_loss(pred.view(self.batch_size,-1), yy.view(self.batch_size, -1))
            train_l2 += l2.item()
            train_mse_local += nn.functional.mse_loss(pred, yy, reduction='mean').item()
        
            self.optimizer.zero_grad()
            l2.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            grads = [param.grad.detach().flatten() for param in self.model.parameters() if param.grad is not None]
            grads_norm = torch.cat(grads).norm()
            self.tensorboard_writer.add_histogram("train/GradNormStep", grads_norm, step)
            grads_norm_epoch += grads_norm
            
            # self.memory.print("After backwardpass")
            
            tepoch.set_postfix({'Batch': step + 1, 'Train l2-loss (in progress)': train_l2,\
                    'Train mse loss (in progress)': train_mse_local})
        
        train_error = train_l2 / len(train_loader)
        train_mse = train_mse_local / len(train_loader)
        avg_loss = train_l2/ nTrain
        grads_norm_epoch = grads_norm_epoch / nTrain
        
        return train_error, train_mse, avg_loss, grads_norm_epoch
                     
    def fno2d_eval(self, val_loader, nVal, val_loss):
        """
        Performs validation for FNO2D recurrent in time model

        Args:
            val_loader (torch.utils.data.DataLoader): validation dataloaders
            nVal: number of validation samples
            val_loss : validation loss

        Returns:
           val_error (float): validation error for one epoch
           val_step_error (float): validation recurrence step error for one epoch
           avg_val_loss (float): average validation error per individual sample
        """
        
        val_l2_step = 0.0
        val_l2_full = 0.0
        avg_val_loss = 0.0
        
        self.model.eval()
        with torch.no_grad():
            for (xx,yy) in (val_loader):
                loss = 0
                xx = xx.to(self.device)
                yy = yy.to(self.device)

                for t in tqdm(range(0, self.T, self.tStep), desc="Validation loop"):
                    y = yy[..., t:t + self.tStep] 
                    im = self.model(xx)
                    loss += val_loss(im.reshape(self.batch_size, -1), y.reshape(self.batch_size, -1))

                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)

                    xx = torch.cat((xx[..., self.tStep:], im), dim=-1)
                    
                val_l2_step += loss.item()
                val_l2_full += val_loss(pred.reshape(self.batch_size, -1), yy.reshape(self.batch_size, -1)).item()
                # self.memory.print("After val first batch")

        val_error = val_l2_full / len(val_loader)
        val_step_error = val_l2_step / len(val_loader) / (self.T / self.tStep)
        avg_val_loss = val_l2_full/ nVal
       
        return val_error, val_step_error, avg_val_loss
    
    def cfno2d_eval(self, val_loader, nVal, val_loss):
        """
        Performs validation for CFNO2D recurrent in time model

        Args:
            val_loader (torch.utils.data.DataLoader): validation dataloaders
            nVal: number of validation samples
            val_loss : validation loss

        Returns:
           val_error (float): validation error for one epoch
           val_step_error (float): validation recurrence step error for one epoch
           avg_val_loss (float): average validation error per individual sample
        """
        
        val_l2_step = 0.0
        val_l2_full = 0.0
        avg_val_loss = 0.0
        
        self.model.eval()
        with torch.no_grad():
            for (xx,yy) in (val_loader):
                loss = 0
                xx = xx.to(self.device)
                yy = yy.to(self.device)

                for t in tqdm(range(0, self.T, self.tStep), desc="Validation loop"):
                    y = yy[..., t:t + self.tStep] 
                    im = self.model(xx)
                    loss += val_loss(im.reshape(self.batch_size, -1), y.reshape(self.batch_size, -1))

                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)

                    xx = torch.cat((xx[..., self.tStep:], im), dim=-1)
                    
                val_l2_step += loss.item()
                val_l2_full += val_loss(pred.reshape(self.batch_size, -1), yy.reshape(self.batch_size, -1)).item()
                # self.memory.print("After val first batch")

        val_error = val_l2_full / len(val_loader)
        val_step_error = val_l2_step / len(val_loader) / (self.T / self.tStep)
        avg_val_loss = val_l2_full/ nVal
       
        return val_error, val_step_error, avg_val_loss
    
    def fno3d_eval(self, val_loader, nVal, val_loss):
        """
        Performs validation for FNO2D recurrent in time  model

        Args:
            val_loader (torch.utils.data.DataLoader): validation dataloaders
            nVal: number of validation samples
            val_loss : validation loss

        Returns:
           val_error (float): validation error for one epoch
           val_mse (float): validation mean squared error for one epoch
           avg_val_loss (float): average validation error per individual sample
        """
        
        val_l2 = 0.0
        val_mse_local = 0.0
        avg_val_loss = 0.0 
        
        self.model.eval()
        with torch.no_grad():
            for (xx,yy) in (val_loader):
                xx = xx.to(self.device)
                yy = yy.to(self.device)
                # yy = UnitGaussianNormalizer(yy).decode(yy)
                pred = self.model(xx).view(self.batch_size, self.nx_state, self.ny, self.T)
                # pred = UnitGaussianNormalizer(pred).decode(pred)

                val_l2 += val_loss(pred.view(self.batch_size, -1), yy.view(self.batch_size, -1)).item()
                val_mse_local += nn.functional.mse_loss(pred, yy, reduction='mean').item()
               
        val_error = val_l2 / len(val_loader)
        val_mse = val_mse_local/ len(val_loader)
        avg_val_loss = val_l2/ nVal
        
        return val_error, val_mse, avg_val_loss
      
    def save_checkpoint(self, epoch:int, train_error:float, 
                    val_error:float, verbose:bool=True
):
        """
        Save torch mdoel checkpoint

        Args:
            epoch (int): epoch
            train_error (float): training error
            val_error (float): validation error
            verbose (bool) : log loss to info.txt filr. Default is True.
        """
        torch.save(
                {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': train_error,
                'val_loss': val_error,
                }, f"{self.save_path}/checkpoint/model_checkpoint_{epoch}.pt")
        
        if verbose:
            with open(f'{self.save_path}/info.txt', 'a') as file:
                            file.write(f"Training loss at {epoch} epoch: {train_error}\n")
                            file.write(f"Validation loss at {epoch} epoch: {val_error}\n")
                            
    def train(
        self,
        save_path:str,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        batch_size:int,
        training_loss,
        val_loss,
        nTrain:int,
        nVal:int,
        tensorboard_writer=None,
        resume_from_checkpoint=None,
    ):
        
        """Trains and validates FNO model

        Args:
            save_path (str): root path to save the model, tensorboard and checkpoint
            train_loader (torch.utils.data.DataLoader): training dataloaders
            val_loader (torch.utils.data.DataLoader): validation dataloaders
            optimizer (torch.optim.Optimizer): training optimizer
            scheduler (torch.optim.lr_scheduler): training learning rate scheduler
            batch_size (int) : training batch size 
            training_loss : training loss
            val_loss : validation loss
            nTrain (int): number of training samples
            nVal (int): number of validation samples
            tensorboard_writer (torch.utils.tensorboard.SummaryWriter): tensorboard logger
            resume_from_checkpoint (str): path to model checkpoint to continue training
        """
        self.save_path = save_path
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size =batch_size
        start_epoch = 0
        if tensorboard_writer is not None:
            self.tensorboard_writer = tensorboard_writer 
        else:
            self.tensorboard_writer = SummaryWriter(log_dir=f"{self.save_path}/tensorboard")
        
        if resume_from_checkpoint is not None:
            self.checkpoint =  torch.load(resume_from_checkpoint)
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            start_epoch = self.checkpoint['epoch']
            start_train_loss = self.checkpoint['train_loss']
            start_val_loss = self.checkpoint['val_loss']
            print(f"Continuing training from {resume_from_checkpoint} at {start_epoch} epoch \
                  with Train-loss {start_train_loss},and Val-loss {start_val_loss}")

        print(f'Starting model training...')
        train_time_start = default_timer()
        for epoch in range(start_epoch, self.epochs):
            with tqdm(unit="batch", disable=False) as tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                t1 = default_timer()
                
                if self.dim == 'FNO2D':
                    train_error, train_step_error, avg_train_loss, grad_norm = self.fno2d_train_single_epoch(tepoch, train_loader, nTrain, training_loss)
                    val_error, val_step_error, avg_val_loss = self.fno2d_eval(val_loader, nVal, val_loss)
                
                    self.tensorboard_writer.add_scalar("train_loss/train_step_l2loss", train_step_error, epoch)
                    self.tensorboard_writer.add_scalar("val_loss/val_step_l2loss", val_step_error, epoch)
                
                elif self.dim == 'CFNO2D':
                    train_error, train_step_error, avg_train_loss, grad_norm = self.cfno2d_train_single_epoch(tepoch, train_loader, nTrain, training_loss)
                    val_error, val_step_error, avg_val_loss = self.cfno2d_eval(val_loader, nVal, val_loss)
                
                    self.tensorboard_writer.add_scalar("train_loss/train_step_l2loss", train_step_error, epoch)
                    self.tensorboard_writer.add_scalar("val_loss/val_step_l2loss", val_step_error, epoch)
                else:
                    train_error, train_mse, avg_train_loss, grad_norm = self.fno3d_train_single_epoch(tepoch, train_loader, nTrain, training_loss)
                    val_error, val_mse, avg_val_loss = self.fno3d_eval(val_loader, nVal, val_loss)
            
                    self.tensorboard_writer.add_scalar("train_loss/train_mseloss", train_mse, epoch)
                    self.tensorboard_writer.add_scalar("val_loss/val_mseloss", val_mse, epoch)
                    
                
                self.tensorboard_writer.add_scalar("train_loss/train_l2loss", train_error, epoch)
                self.tensorboard_writer.add_scalar("train_loss/avg_trainloss", avg_train_loss, epoch)
                self.tensorboard_writer.add_scalar("train/GradNormEpoch", grad_norm, epoch)
                self.tensorboard_writer.add_scalar("val_loss/val_l2loss", val_error, epoch)
                self.tensorboard_writer.add_scalar("val_loss/avg_valloss", avg_val_loss, epoch)
                         
                t2 = default_timer()
                tepoch.set_postfix({ \
                    'Epoch': epoch, \
                    'Time per epoch (s)': (t2-t1), \
                    'Train l2loss': train_error,\
                    'Val l2loss':  val_error
                    })
                
            tepoch.close()
            
            if epoch > 0 and (epoch % 10 == 0 or epoch == self.epochs-1):
                self.save_checkpoint(epoch, train_error, val_error, verbose=True)
                
            if self.exit_signal_handler:
                signal_handler = get_signal_handler()
                if any(signal_handler.signals_received()):
                    self.save_checkpoint(epoch, train_error, val_error, verbose=True)
                    print('exiting program after receiving SIGTERM.')
                    train_time_stop = default_timer()
                    print(f'Total training+validation time (s): {train_time_stop - train_time_start}')
                    sys.exit()
            
            if self.exit_duration_in_mins is not None:
                train_time = (time.time() - _TRAIN_START_TIME) / 60.0
                done_check = torch.tensor(
                    [train_time > self.exit_duration_in_mins],
                    dtype=torch.int, device=self.device)
                done = done_check.item()
                if done:
                    self.save_checkpoint(epoch, train_error, val_error, verbose=True)
                    print('exiting program after {} minutes'.format(train_time))
                    sys.exit()
        
        train_time_stop = default_timer()
        print(f'Exiting train()...')
        print(f'Total training+validation time (s): {train_time_stop - train_time_start}')
