import os
import sys
import functools
import operator
import h5py
import math
import copy
import scipy
import pickle
import scipy.io
import argparse
import time
import numpy as np
from importlib import reload
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from timeit import default_timer
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from utils import CudaMemoryDebugger, format_tensor_size, LpLoss, get_signal_handler, _set_signal_handler

_GLOBAL_SIGNAL_HANDLER = None
_TRAIN_START_TIME = time.time()

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1              #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :,  :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = nn.functional.gelu(x)
        x = self.mlp2(x)
        return x

class FNO2d(nn.Module):
    memory = CudaMemoryDebugger(print_mem=True)
    
    def __init__(self, modes1, modes2, width, T_in, T):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=4*sizex//xStep, z=sizez/zStep, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=4*sizex//xStep, z=sizez//zStep, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.T_in = T_in
        self.T = T
        self.padding = 8 # pad the domain if input is non-periodic

        self.p = nn.Linear(self.T_in+2, self.width) # input channel is T_in+2: the solution of the previous T_in timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        #memory.print("after p(x)")
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = nn.functional.gelu(x)
        #memory.print("after FNO1")

        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = nn.functional.gelu(x)
        #memory.print("after FNO2")
        
        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = nn.functional.gelu(x)
        #memory.print("after FNO3")
        
        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        #memory.print("after FNO4")
        
        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        #memory.print("after q(x)")
        x = x.permute(0, 2, 3, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    
    def print_size(self):
        properties = []

        for param in self.parameters():
            properties.append([list(param.size()+(2,) if param.is_complex() else param.size()), param.numel(), (param.data.element_size() * param.numel())/1000])
            
        elementFrame = pd.DataFrame(properties, columns = ['ParamSize', 'NParams', 'Memory(KB)'])
 
        print(f'Total number of model parameters: {elementFrame["NParams"].sum()} with (~{format_tensor_size(elementFrame["Memory(KB)"].sum()*1000)})')
        return elementFrame

def multi_data(reader, start_time, end_time, timestep, samples):
    a = []
    u = []
    for start_index in range(start_time, end_time, timestep):
        a.append(torch.tensor(reader[task][:samples, ::xStep, ::zStep, start_index: start_index + (T_in*tStep): tStep], dtype=torch.float))
        u.append(torch.tensor(reader[task][:samples, ::xStep, ::zStep, start_index + (T_in*tStep):  start_index + (T_in + T)*tStep: tStep], dtype=torch.float))
    a = torch.stack(a)
    u = torch.stack(u)
    
    a_multi = a.reshape(a.shape[0]*a.shape[1], a.shape[2], a.shape[3], a.shape[4])
    u_multi = u.reshape(u.shape[0]*u.shape[1], u.shape[2], u.shape[3], u.shape[4])
    
    return a_multi, u_multi

def train(args):  
    ## config
    modes = 12
    width = 20

    batch_size = 5
    learning_rate = 0.00039
    weight_decay = 1e-05
    scheduler_step = 100.0
    scheduler_gamma = 0.98
    
    epochs = 3000
    iterations = epochs*(ntrain//batch_size)

    gridx = 4*256
    gridz = 64

    ## load data
    if args.single_data_path is not None:
        train_data_path = val_data_path = test_data_path = args.single_data_path
        train_reader = val_reader = test_reader = h5py.File(train_data_path, mode="r")
    else:  
        train_data_path = args.train_data_path
        train_reader = h5py.File(train_data_path, mode="r")
        val_data_path = args.val_data_path
        val_reader = h5py.File(val_data_path, mode="r")
        # test_data_path = args.test_data_path
        # test_reader = h5py.File(test_data_path, mode="r")
    
    if args.multi_step:
        train_a, train_u = multi_data(train_reader, start_index, stop_index, timestep, ntrain)
        val_a, val_u = multi_data(val_reader, start_index, stop_index, timestep, nval)
    else:
        train_a = torch.tensor(train_reader['train'][:ntrain, ::xStep, ::zStep, start_index: start_index + (T_in*tStep): tStep],dtype=torch.float)
        train_u = torch.tensor(train_reader['train'][:ntrain, ::xStep, ::zStep, start_index + (T_in*tStep):  start_index + (T_in + T)*tStep: tStep], dtype=torch.float)

        val_a = torch.tensor(val_reader['val'][:nval, ::xStep, ::zStep, start_index: start_index + (T_in*tStep): tStep],dtype=torch.float)
        val_u = torch.tensor(val_reader['val'][:nval, ::xStep, ::zStep, start_index + (T_in*tStep):  start_index + (T_in + T)*tStep: tStep],dtype=torch.float)

    print(f"Train data:{train_u.shape}")
    print(f"Val data:{val_u.shape}")
    assert (gridx == train_u.shape[-3])
    assert (gridz == train_u.shape[-2])
    assert (T == train_u.shape[-1])

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_a, val_u), batch_size=batch_size, shuffle=False)
    # test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

    run = args.run
    
    fno_path = Path(f'{args.model_save_path}/rbc_fno2d_time_N{ntrain}_epoch{epochs}_m{modes}_w{width}_bs{batch_size}_run{run}')
    fno_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(f'{fno_path}/checkpoint')
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=f"{fno_path}/tensorboard")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    torch.cuda.empty_cache()
    memory = CudaMemoryDebugger(print_mem=True)

    ################################################################
    # training and evaluation
    ################################################################

    model2d = FNO2d(modes, modes, width, T_in, T).to(device)
    # print(model2d.print_size())
    n_params = model2d.print_size()
    # memory.print("after intialization")

    optimizer = torch.optim.Adam(model2d.parameters(), lr=learning_rate, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    # optimizer = torch.optim.AdamW(model2d.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    myloss = LpLoss(size_average=False)
    start_epoch = 0

    print(f"fno_modes={modes}\n \
            layer_width={width}\n \
            (Tin,Tout):{T_in,T}\n \
            batch_size={batch_size}\n \
            optimizer={optimizer}\n \
            lr_scheduler={scheduler}\n \
            lr_scheduler_step={scheduler_step}\n \
            lr_scheduler_gamma={scheduler_gamma}\n\
            fno_path={fno_path}")
    
    with open(f'{fno_path}/info.txt', 'a') as file:
        file.write("-------------------------------------------------\n")
        file.write(f"Model Card for FNO-2D with (x,z) and Recurrent in Time\n")
        file.write("-------------------------------------------------\n")
        file.write(f"model_params:{n_params}\n")
        file.write(f"fno_modes:{modes}\n")
        file.write(f"layer_widths:{width}\n")
        file.write(f"(Tin,Tout):{T_in,T}\n")
        file.write(f"(ntrain, nval, ntest): {ntrain, nval, ntest}\n")
        file.write(f"batch_size: {batch_size}\n")
        file.write(f"optimizer: {optimizer}\n")
        file.write(f"lr_scheduler: {scheduler}\n")
        file.write(f"lr_scheduler_step: {scheduler_step}\n")
        file.write(f"lr_scheduler_gamma: {scheduler_gamma}\n")
        file.write(f"input_time_steps: {T_in}\n")
        file.write(f"output_time_steps: {T}\n")
        file.write(f"time_start_index: {start_index}\n")
        file.write(f"(x_step, z_step, t_step): {xStep, zStep, tStep}\n")
        file.write(f"grid(x,z): ({gridx,gridz})\n")
        file.write(f"train_data (a,u): {train_a.shape, train_u.shape}\n")
        file.write(f"model_path: {fno_path}\n")
        file.write("-------------------------------------------------\n")

    if args.load_checkpoint:
        checkpoint = torch.load(args.checkpoint_path)
        model2d.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_train_loss = checkpoint['train_loss']
        start_val_loss = checkpoint['val_loss']
        print(f"Continuing training from {checkpoint_path} at {start_epoch} with Train-L2-loss {start_train_loss},\
                and Val-L2-loss {start_val_loss}")
        
    for epoch in range(start_epoch, epochs):
        with tqdm(unit="batch", disable=False) as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            model2d.train()
            #memory.print("After model2d.train()")
            
            t1 = default_timer()
            train_l2_step = 0
            train_l2_full = 0
            
            for step, (xx, yy) in enumerate(train_loader):
                loss = 0
                xx = xx.to(device)
                yy = yy.to(device)
                #memory.print("After loading first batch")

                for t in tqdm(range(0, T, tStep), desc="Train loop"):
                    y = yy[..., t:t + tStep]
                    im = model2d(xx)
                    loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                        
                    # print(f"{t}: y={y.shape},x={xx.shape},pred={pred.shape}")
                    xx = torch.cat((xx[..., tStep:], im), dim=-1)
                    # print(f"{t}: new_xx={xx.shape}")

                train_l2_step += loss.item()
                l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
                train_l2_full += l2_full.item()

                optimizer.zero_grad()
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                grads = [param.grad.detach().flatten() for param in model2d.parameters()if param.grad is not None]
                grads_norm = torch.cat(grads).norm()
                writer.add_histogram("train/GradNormStep",grads_norm, step)
                
                #memory.print("After backwardpass")
                
                tepoch.set_postfix({'Batch': step + 1, 'Train l2 loss (in progress)': train_l2_full,\
                        'Train l2 step loss (in progress)': train_l2_step})
            
        
            train_error = train_l2_full / ntrain
            train_step_error = train_l2_step / ntrain / (T / tStep)
            writer.add_scalar("train_loss/train_l2loss", train_error, epoch)
            writer.add_scalar("train_loss/train_step_l2loss", train_step_error, epoch)
            writer.add_scalar("train/GradNorm", grads_norm, epoch)
            
            val_l2_step = 0
            val_l2_full = 0
            model2d.eval()
            with torch.no_grad():
                for step, (xx,yy) in enumerate(val_loader):
                    loss = 0
                    xx = xx.to(device)
                    yy = yy.to(device)

                    for t in tqdm(range(0, T, tStep), desc="Validation loop"):
                        y = yy[..., t:t + tStep]
                        im = model2d(xx)
                        loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                        if t == 0:
                            pred = im
                        else:
                            pred = torch.cat((pred, im), -1)

                        xx = torch.cat((xx[..., tStep:], im), dim=-1)
                        
                    val_l2_step += loss.item()
                    val_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
                    #memory.print("After val first batch")

            val_error = val_l2_full / nval
            val_step_error = val_l2_step / nval / (T / tStep)
            writer.add_scalar("val_loss/val_step_l2loss", val_error, epoch)
            writer.add_scalar("val_loss/val_l2loss", val_step_error, epoch)
                
            t2 = default_timer()
            tepoch.set_postfix({ \
                'Epoch': epoch, \
                'Time per epoch (s)': (t2-t1), \
                'Train l2loss': train_error,\
                'Val l2loss':  val_error 
                })
            
        tepoch.close()
        
        if epoch > 0 and (epoch % 100 == 0 or epoch == epochs-1):
            with open(f'{fno_path}/info.txt', 'a') as file:
                    file.write(f"Training Error at {epoch}: {train_error}\n")
                    file.write(f"Validation Error at {epoch}: {val_error}\n")
            torch.save(
                {
                'epoch': epoch,
                'model_state_dict': model2d.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_error,
                'val_loss': val_error,
                }, f"{checkpoint_path}/model_checkpoint_{epoch}.pt")
            
        if args.exit_signal_handler:
            signal_handler = get_signal_handler()
            if any(signal_handler.signals_received()):
                with open(f'{fno_path}/info.txt', 'a') as file:
                    file.write(f"Training Error at {epoch}: {train_error}\n")
                    file.write(f"Validation Error at {epoch}: {val_error}\n")
                torch.save(
                    {
                    'epoch': epoch,
                    'model_state_dict': model2d.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_error,
                    'val_loss': val_error,
                    }, f"{checkpoint_path}/model_checkpoint_{epoch}.pt")
                print('exiting program after receiving SIGTERM.')
        
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_check = torch.tensor(
                [train_time > args.exit_duration_in_mins],
                dtype=torch.int, device=device)
            done = done_check.item()
            if done:
                with open(f'{fno_path}/info.txt', 'a') as file:
                    file.write(f"Training Error at {epoch}: {train_error}\n")
                    file.write(f"Validation Error at {epoch}: {val_error}\n")
                torch.save(
                    {
                    'epoch': epoch,
                    'model_state_dict': model2d.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_error,
                    'val_loss': val_error,
                    }, f"{checkpoint_path}/model_checkpoint_{epoch}.pt")
                print('exiting program after {} minutes'.format(train_time))
          
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FNO Training')
    parser.add_argument('--run', type=int, default=1,
                        help='training tracking number')
    parser.add_argument('--model_save_path', type=str,default=os.getcwd(),
                        help='path to which FNO model is saved')
    parser.add_argument('--single_data_path', type=str,default=None,
                        help='path to hdf5 file containing train, val and test data')
    parser.add_argument('--train_data_path', type=str,
                        help='path to train data hdf5 file')
    parser.add_argument('--val_data_path', type=str,
                        help='path to validation data hdf5 file')
    parser.add_argument('--test_data_path', type=str,
                        help='path to test data hdf5 file')
    parser.add_argument('--load_checkpoint', action="store_true",
                        help='load checkpoint')
    parser.add_argument('--multi_step', action="store_true",
                        help='take multiple step data')
    parser.add_argument('--checkpoint_path', type=str,
                        help='folder containing checkpoint')
    parser.add_argument('--exit-signal-handler', action='store_true',
                       help='Dynamically save the checkpoint and shutdown the '
                       'training if SIGTERM is received')
    parser.add_argument('--exit-duration-in-mins', type=int, default=None,
                       help='Exit the program after this many minutes.')
    args = parser.parse_args()
    
    if args.exit_signal_handler:
        _set_signal_handler()
    
    ntrain = 100
    nval = 50
    xStep = 1
    zStep = 1
    tStep = 1
    T_in = 1
    T = 10
    start_index = 0
    stop_index = 990
    timestep = 100
    
    train(args)