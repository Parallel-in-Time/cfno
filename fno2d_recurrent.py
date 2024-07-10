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
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from utils import CudaMemoryDebugger, format_tensor_size, LpLoss

parser = argparse.ArgumentParser(description='FNO Training')
parser.add_argument('--model_save_path', type=str,default=os.getcwd(),
                    help='path to which FNO model is saved')
parser.add_argument('--load_checkpoint', action="store_true",
                    help='load checkpoint')
parser.add_argument('--data_path', type=str,
                    help='path to data')
parser.add_argument('--checkpoint_path', type=str,
                    help='folder containing checkpoint')
args = parser.parse_args()

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
    
    def __init__(self, modes1, modes2, width):
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
        self.padding = 8 # pad the domain if input is non-periodic

        self.p = nn.Linear(12, self.width) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
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
        # memory.print("after p(x)")
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = nn.functional.gelu(x)
        # memory.print("after FNO1")

        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = nn.functional.gelu(x)
        # memory.print("after FNO2")
        
        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = nn.functional.gelu(x)
        # memory.print("after FNO3")
        
        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        # memory.print("after FNO4")
        
        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        # memory.print("after q(x)")
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
    
## config
ntrain = 100
ntest = 50
nval = 50

modes = 12
width = 20

batch_size = 20
learning_rate = 0.001
epochs = 500
iterations = epochs*(ntrain//batch_size)

fno_path = Path(f'{args.model_save_path}/rbc_fno_2d_time_N{ntrain}_epoch{epochs}_m{modes}_w{width}')
fno_path.mkdir(parents=True, exist_ok=True)

checkpoint_path = Path(f'{fno_path}/checkpoint')
checkpoint_path.mkdir(parents=True, exist_ok=True)

writer = SummaryWriter(log_dir=f"{fno_path}/tensorboard")

gridx = 4*256
gridz = 64

xStep = 1
zStep = 1
tStep = 1

start_index = 500
T_in = 10
T = 10

## load data
ntrain = 100
ntest = 50
nval = 50
data_path = args.data_path
reader = h5py.File(data_path, mode="r")
train_a = torch.tensor(reader['train'][:ntrain, ::xStep, ::zStep, start_index: start_index+T_in],dtype=torch.float)
train_u = torch.tensor(reader['train'][:ntrain, ::xStep, ::zStep, start_index+T_in:T+start_index+T_in], dtype=torch.float)

test_a = torch.tensor(reader['test'][:ntest, ::xStep, ::zStep, start_index: start_index+T_in],dtype=torch.float)
test_u = torch.tensor(reader['test'][:ntest, ::xStep, ::zStep, start_index+T_in:T+start_index+T_in],dtype=torch.float)

val_a = torch.tensor(reader['val'][:nval, ::xStep, ::zStep, start_index: start_index+T_in],dtype=torch.float)
val_u = torch.tensor(reader['val'][:nval, ::xStep, ::zStep, start_index+T_in:T+start_index+T_in],dtype=torch.float)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_a, val_u), batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")
torch.cuda.empty_cache()
memory = CudaMemoryDebugger(print_mem=True)

################################################################
# training and evaluation
################################################################

model2d = FNO2d(modes, modes, width).to(device)
# print(model2d.print_size())
n_params = model2d.print_size()
memory.print("after intialization")

optimizer = torch.optim.Adam(model2d.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = LpLoss(size_average=False)
start_epoch = 0

if args.load_checkpoint:
    checkpoint = torch.load(args.checkpoint_path)
    model2d.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint['epoch']
    print(f"Continuing training from {checkpoint_path} at {start_epoch}")
    
for epoch in range(start_epoch, epochs):
    with tqdm(unit="batch", disable=False) as tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        model2d.train()
        # memory.print("after model.train()")
        
        t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0
        
        for step, (xx, yy) in enumerate(train_loader):
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            # memory.print("after loading first batch")

            for t in tqdm(range(0,T, tStep), desc="Train loop"):
                y = yy[..., t:t + tStep]
                im = model2d(xx)
                # print("ouput:",y.shape,"pred:", im.shape)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., tStep:], im), dim=-1)

            train_l2_step += loss.item()
            l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
            train_l2_full += l2_full.item()

            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            
            grads = [param.grad.detach().flatten() for param in model2d.parameters()if param.grad is not None]
            grads_norm = torch.cat(grads).norm()
            writer.add_histogram("train/GradNormStep",grads_norm, step)
            
            scheduler.step()
            # memory.print("after backwardpass")
            
            tepoch.set_postfix({'Batch': step + 1, 'Train l2 loss (in progress)': train_l2_full,\
                     'Train l2 step loss (in progress)': train_l2_step})

        
        writer.add_scalar("train_loss/train_l2loss", train_l2_full, epoch)
        writer.add_scalar("train_loss/train_step_l2loss", train_l2_step, epoch)
        writer.add_scalar("train/GradNorm", grads_norm, epoch)
        
        val_l2_step = 0
        val_l2_full = 0
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
                # memory.print("after val first batch")

            writer.add_scalar("val_loss/val_step_l2loss", val_l2_step, epoch)
            writer.add_scalar("val_loss/val_l2loss", val_l2_full, epoch)
            
        t2 = default_timer()
        train_error = train_l2_full / ntrain
        val_error = val_l2_full / nval
        tepoch.set_postfix({ \
            'Epoch': epoch, \
            'Time per epoch': (t2-t1), \
            'Train l2loss step': train_l2_step / ntrain / (T / tStep),\
            'Train l2loss': train_error,\
            'Val l2loss step': val_l2_step / nval / (T / tStep), \
            'Val l2loss':  val_error 
            })
    tepoch.close()
    
    with open(f'{fno_path}/errors.txt', 'w') as file:
        file.write("Training Error: " + str(train_error) + "\n")
        file.write("Validation Error: " + str(val_error) + "\n")
        file.write("Current Epoch: " + str(epoch) + "\n")
        # file.write("Params: " + str(n_params) + "\n")
  
            
    if epoch % 100 == 0 or epoch == epochs-1:
        torch.save(model2d.state_dict(), f"{checkpoint_path}/model_checkpoint_{epoch}.pt")