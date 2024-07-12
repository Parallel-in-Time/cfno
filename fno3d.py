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
from torchsummary import summary    # TODO: is it really needed ?
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from utils import CudaMemoryDebugger, format_tensor_size, LpLoss, UnitGaussianNormalizer



class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels  
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        # k_max = 12 in paper 
        self.modes1 = modes1                 
        self.modes2 = modes2
        self.modes3 = modes3
        
        # R
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        # summation along in_channel 
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        # x = [batchsize, width, size_x, size_y, T + padding]
        batchsize = x.shape[0]
        
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1]) 
        # [batchsize, width, size_x, size_y, if (T + padding) is even ((T + padding)/2 +1) else (T + padding)/2 ]
        
        # Multiply relevant Fourier modes (Corners of R) --> R.FFT(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)  # upper right
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2) # upper left
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3) # lower right
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4) # lower left

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1))) # x = [batchsize, width, size_x, size_y, T + padding]
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        # input: [batchsize, in_channel=width, size_x, size_y, T + padding]
        # weight: [mid_channel=width, in_channel=width, 1,1,1]
        # output: [batchsize, out_channel=mid_channel, size_x, size_y, T + padding]
        x = self.mlp1(x)
        x = nn.functional.gelu(x)
        # output: [batchsize, out_channel=mid_channel, size_x, size_y, T + padding]
        x = self.mlp2(x)
        # input: [batchsize, in_channel=mid_channel, size_x, size_y, T + padding]
        # weight: [out_channel=width, mid_channel=width, 1, 1, 1]
        # output: [batchsize, out_channel=width, size_x, size_y, T + padding]
        return x

class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t).
        It's a constant function in time, except for the last index.
        input shape: (batchsize,  x=sizex, y=sizey, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=sizex, y=sizey, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        
        # x = (batchsize,   x=sizex, y=sizey, t=40, c=13)
        # input channel is 13: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)
        self.p = nn.Linear(13, self.width)
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device) # [batchsize,   size_x, size_y, T, c=T_in] ---> [batchsize,   size_x, size_y, T, c=3]
        x = torch.cat((x, grid), dim=-1)        # [batchsize,   size_x, size_y, T, c=T_in+3]
        x = self.p(x)                           
        # input: [batchsize,   size_x, size_y, T, c=T_in+3], 
        # Weight: [width,T_in+3]
        # Output: [batchsize,   size_x, size_y, T, c=width]
        
        x = x.permute(0, 4, 1, 2, 3)           # [batchsize,  size_x, size_y, T, c=width] --> [batchsize, width,   size_x, size_y, T]
        x = nn.functional.pad(x, [0,self.padding]) # pad the domain if input is non-periodic, padded along last dim of x
        
        # padding order:(padding_left,padding_right, 
        #                 padding_top,padding_bottom,
        #                 padding_front,padding_back)
                
        # [batchsize, width,   size_x, size_y, T + padding]
        
        x1 = self.conv0(x) # SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        
        # input: [batchsize, width,   size_x, size_y, T + padding]
        # weight: torch.rand(in_channels=width, out_channels=width, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)
        # Output: [batchsize, out_channel=width,   size_x, size_y, T + padding]
        
        x1 = self.mlp0(x1) # MLP(self.width, self.width, self.width)
        # input: [batchsize, in_channel=width,   size_x, size_y, T + padding]
        # output: [batchsize, out_channel=width,    size_x, size_y, T + padding]
        
        x2 = self.w0(x)   # nn.Conv3d(self.width, self.width, 1)
        # input: [batchsize, in_channel=width,   size_x, size_y, T + padding]
        # weight: [out_channel=width, in_channel=width, 1, 1,1]
        # output: [batchsize, out_channel=width,   size_x, size_y, T + padding]
        
        x = x1 + x2
        x = nn.functional.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = nn.functional.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = nn.functional.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        # output: [batchsize, out_channel=width,   size_x, size_y, T + padding]
        
        x = x[..., :-self.padding]
        # output: [batchsize, out_channel=width,   size_x, size_y, T]
        
        x = self.q(x) # MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)
        
        # input: [batchsize, in_channel=width,  size_x, size_y, T ]
        # weight: [mid_channel=4*width, in_channel=width, 1,1,1]
        # output: [batchsize, out_channel=mid_channel=4*width,   size_x, size_y, T ]
        # x = self.mlp1(x)
        # x = torch.nn.Functional.gelu(x)
        # output: [batchsize, out_channel=mid_channel=4*width,  size_x, size_y, T]
        # x = self.mlp2(x)
        # input: [batchsize, in_channel=mid_channel=4*width,  size_x, size_y, T]
        # weight: [out_channel=1, mid_channel=4*width, 1, 1, 1]
        # output: [batchsize, out_channel=1,   size_x, size_y, T]
        
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        # output: [batchsize,   size_x, size_y, T, out_channel=1]
        return x


   
    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)# [batchsize,  size_x, size_y, T, 3]
    
    def print_size(self):
        properties = []

        for param in self.parameters():
            properties.append([list(param.size()+(2,) if param.is_complex() else param.size()), param.numel(), (param.data.element_size() * param.numel())/1000])
            
        elementFrame = pd.DataFrame(properties, columns = ['ParamSize', 'NParams', 'Memory(KB)'])
 
        print(f'Total number of model parameters: {elementFrame["NParams"].sum()} with (~{format_tensor_size(elementFrame["Memory(KB)"].sum()*1000)})')
        return elementFrame
    

def train(args):
    ## config
    ntrain = 100
    nval = 50
    ntest = 50

    modes = 8
    width = 20

    batch_size = 5
    learning_rate = 0.001
    weight_decay = 1e-4
    scheduler_step = 10.0
    scheduler_gamma = 0.98

    epochs = 500
    iterations = epochs*(ntrain//batch_size)

    fno_path = Path(f'{args.model_save_path}/rbc_fno_3d_N{ntrain}_epoch{epochs}_m{modes}_w{width}_bs{batch_size}')
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
    data_path = args.data_path
    reader = h5py.File(data_path, mode="r")
    train_a = torch.tensor(reader['train'][:ntrain, ::xStep, ::zStep, start_index: start_index+T_in],dtype=torch.float)
    train_u = torch.tensor(reader['train'][:ntrain, ::xStep, ::zStep, start_index+T_in:T+start_index+T_in], dtype=torch.float)

    val_a = torch.tensor(reader['val'][:nval, ::xStep, ::zStep, start_index: start_index+T_in],dtype=torch.float)
    val_u = torch.tensor(reader['val'][:nval, ::xStep, ::zStep, start_index+T_in:T+start_index+T_in],dtype=torch.float)

    test_a = torch.tensor(reader['test'][:ntest, ::xStep, ::zStep, start_index: start_index+T_in],dtype=torch.float)
    test_u = torch.tensor(reader['test'][:ntest, ::xStep, ::zStep, start_index+T_in:T+start_index+T_in],dtype=torch.float)

    print(f"Train data:{train_u.shape}")
    print(f"Val data:{val_u.shape}")

    a_normalizer = UnitGaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    val_a = a_normalizer.encode(val_a)

    y_normalizer = UnitGaussianNormalizer(train_u)
    train_u = y_normalizer.encode(train_u)
    val_u = y_normalizer.encode(val_u)

    train_a = train_a.reshape(ntrain, gridx, gridz, 1, T_in).repeat([1,1,1,T,1])
    val_a = val_a.reshape(nval, gridx, gridz, 1, T_in).repeat([1,1,1,T,1])

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_a, val_u), batch_size=batch_size, shuffle=False)
    # test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    torch.cuda.empty_cache()
    memory = CudaMemoryDebugger(print_mem=True)

    ################################################################
    # training and evaluation
    ################################################################

    model3d = FNO3d(modes, modes, modes, width).to(device)
    # print(model3d.print_size())
    n_params = model3d.print_size()
    memory.print("after intialization")

    optimizer = torch.optim.Adam(model3d.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    # optimizer = torch.optim.AdamW(model3d.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    
    y_normalizer.to(device)
    myloss = LpLoss(size_average=False)
    start_epoch = 0


    print(f"bs={batch_size}; \
            lr={learning_rate}; \
            weight_decay={weight_decay}; \
            scheduler_step={scheduler_step};\
            scheduler_gamma={scheduler_gamma};\
            fno_path={fno_path}")

    if args.load_checkpoint:
        checkpoint = torch.load(args.checkpoint_path)
        model3d.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        print(f"Continuing training from {checkpoint_path} at {start_epoch}")
        
    for epoch in range(start_epoch, epochs):
        with tqdm(unit="batch", disable=False) as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            model3d.train()
            memory.print("after model3d.train()")
            
            t1 = default_timer()
            train_mse = 0
            train_l2 = 0
            
            for step, (xx, yy) in enumerate(train_loader):
                xx = xx.to(device)
                yy = yy.to(device)
                memory.print("after loading first batch")

                pred = model3d(xx).view(batch_size, gridx, gridz, T)
                mse = nn.functional.mse_loss(pred, yy, reduction='mean')
                
                print(f"{step} with encoding: y={yy.shape},x={xx.shape},pred={pred.shape}")
                
                yy = y_normalizer.decode(yy)
                pred = y_normalizer.decode(pred)
                
                print(f"{step} decoded: y={yy.shape},x={xx.shape},pred={pred.shape}")
                
                l2 = myloss(pred.view(batch_size,-1), yy.view(batch_size, -1))
                
                optimizer.zero_grad()
                l2.backward()
                optimizer.step()
                scheduler.step()
                
                train_mse += mse.item()
                train_l2 += l2.item()
                
                grads = [param.grad.detach().flatten() for param in model3d.parameters()if param.grad is not None]
                grads_norm = torch.cat(grads).norm()
                writer.add_histogram("train/GradNormStep",grads_norm, step)
                
                memory.print("after backwardpass")
                
                tepoch.set_postfix({'Batch': step + 1, 'Train l2 loss (in progress)': train_l2,\
                        'Train mse loss (in progress)': train_mse})
            
            # scheduler.step()
            train_l2_error = train_l2 / ntrain
            train_mse_error = train_mse / ntrain 
            writer.add_scalar("train_loss/train_l2loss", train_l2_error, epoch)
            writer.add_scalar("train_loss/train_mseloss", train_mse_error, epoch)
            writer.add_scalar("train/GradNorm", grads_norm, epoch)
            
            val_l2 = 0
            model3d.eval()
            with torch.no_grad():
                for step, (xx,yy) in enumerate(val_loader):
                    xx = xx.to(device)
                    yy = yy.to(device)
                    memory.print("after val first batch")
                    
                    pred = model3d(xx).view(batch_size, gridx, gridz, T)
                    pred = y_normalizer.decode(pred)
                    
                    val_l2 += myloss(pred.view(batch_size,-1), yy.view(batch_size, -1)).item()
                    

            val_l2_error = val_l2 / nval
            writer.add_scalar("val_loss/val_l2loss", val_l2_error, epoch)
                
            t2 = default_timer()
            tepoch.set_postfix({ \
                'Epoch': epoch, \
                'Time per epoch': (t2-t1), \
                'Train l2loss': train_l2_error ,\
                'Train mseloss': train_mse_error,\
                'Val l2loss':  val_l2_error 
                })
            
        tepoch.close()
        
        with open(f'{fno_path}/info.txt', 'w') as file:
            file.write("Training Error: " + str(train_l2_error) + "\n")
            file.write("Validation Error: " + str(val_l2_error) + "\n")
            file.write("Current Epoch: " + str(epoch) + "\n")
            file.write("batch_size:" + str(batch_size) + "\n")
            file.write("learning_rate:" + str(learning_rate)+ "\n")
            # file.write("Params: " + str(n_params) + "\n")
    
            
        if epoch % 100 == 0 or epoch == epochs-1:
            torch.save(model3d.state_dict(), f"{checkpoint_path}/model_checkpoint_{epoch}.pt")
        
if __name__ == '__main__':
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
    
    train(args)