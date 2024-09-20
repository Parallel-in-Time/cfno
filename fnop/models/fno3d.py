
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fnop.utils import CudaMemoryDebugger, format_tensor_size
from fnop.layers.mlp import MLP
from fnop.layers.spectral_layers import SpectralConv3d

class FNO3D(nn.Module):
    """
    FNO3D model: 3D space-time convolutions 

    Args:
        modes1, modes2, modes3 (int): fourier modes
        width (int): width of layer
        T_in (int): number of input timesteps
        T (int): number of output timesteps
        n_dim (int): convolution dimension. Default is 3.
    """
    def __init__(self,
                 modes1:int, 
                 modes2:int,
                 modes3:int, 
                 width:int, 
                 T_in:int, 
                 T:int,
                 n_dim:int=3,
    
    ):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.p
        2. 4 layers of the integral operators u' = (W + K)(u)
            W defined by self.w; K defined by self.conv + self.mlp
        3. Project from the channel space to the output space by self.q
        
        input: the solution of the first T_in timesteps + 3 locations (u(1, x, y), ..., u(T_in, x, y), x, y, t).
        It's a constant function in time, except for the last index.
        input shape: (batchsize, x=4*sizex//xStep, y=size_y/yStep, t=T, c=T_in+3)
        output: the solution of the next T timesteps
        output shape: (batchsize, x=4*sizex//xStep, y=size_y/yStep, t=T, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.T_in = T_in
        self.T = T
        self.n_dim = n_dim
        padding_est = (2 * self.modes3) - self.T   # pad the domain if input is non-periodic
        if (self.T + padding_est) % 2 == 0:
            self.padding = padding_est - 1
        else:
            self.padding = padding_est
        
        print(f"Padding: {self.padding}")
        print(f"ndim: {self.n_dim}")
        
        # x = (batchsize, x=sizex, y=size_y, t=T, c=T_in+n_dim)
        # input channel is T_in+n_dim: the solution of the T_in timesteps + n_dim locations 
        # (u(t, x, y), ..., u(t+T_in, x, y), x, y, t)
        self.p = nn.Linear(self.T_in+self.n_dim, self.width)
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.mlp0 = MLP(self.width, self.width, self.width, self.n_dim)
        self.mlp1 = MLP(self.width, self.width, self.width, self.n_dim)
        self.mlp2 = MLP(self.width, self.width, self.width, self.n_dim)
        self.mlp3 = MLP(self.width, self.width, self.width, self.n_dim)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width * 4, self.n_dim) 

        self.memory = CudaMemoryDebugger(print_mem=True)
        
    def forward(self, x):
        grid = self.get_grid(x.shape, x.device) 
        x = torch.cat((x, grid), dim=-1)        
        # [batchsize, size_x, size_y, T, c=T_in] ---> [batchsize, size_x, size_y, T, c=T_in+n_dim]
        
        x = self.p(x)     
        # nn.Linear(self.T_in+n_dim, self.width)                      
            # input: [batchsize, size_x, size_y, T, c=T_in+n_dim], 
            # Weight: [width,T_in+n_dim]
            # Output: [batchsize, size_x, size_y, T, c=width]
        
        # self.memory.print("after p(x)")
        
        x = x.permute(0, 4, 1, 2, 3)           
        # [batchsize, size_x, size_y, T, c=width] ---> [batchsize, width, size_x, size_y, T]
       
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        # padding order:(padding_left,padding_right, 
        #                 padding_top,padding_bottom,
        #                 padding_front,padding_back)
        x = nn.functional.pad(x, [0,self.padding]) # pad the domain if input is non-periodic, padded along last dim of x
        # [batchsize, width, size_x, size_y, T + padding]
        
        x1 = self.conv0(x) 
        # SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
            # input: [batchsize, width, size_x, size_y, T + padding]
            # weight: torch.rand(in_channels=width, out_channels=width, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)
            # Output: [batchsize, out_channel=width, size_x, size_y, T + padding]
            
        x1 = self.mlp0(x1) 
        # MLP(self.width, self.width, self.width)
            # input: [batchsize, in_channel=width, size_x, size_y, T + padding]
            # weight: [mid_channel=width, in_channel=width, 1,1,1]
            # output: [batchsize, out_channel=mid_channel, size_x, size_y, T + padding]
            # x = nn.functional.gelu(x)
            # input: [batchsize, mid_channel, size_x, size_y, T + padding]
            # output: [batchsize, mid_channel, size_x, size_y, T + padding]
            # x = self.mlp2(x)
            # input: [batchsize, in_channel=mid_channel, size_x, size_y, T + padding]
            # weight: [out_channel=width, mid_channel=width, 1, 1, 1]
            # output: [batchsize, out_channel=width, size_x, size_y, T + padding]
           
        
        x2 = self.w0(x) 
        # nn.Conv3d(self.width, self.width, 1)
            # input: [batchsize, in_channel=width, size_x, size_y, T + padding]
            # weight: [out_channel=width, in_channel=width, 1, 1,1]
            # output: [batchsize, out_channel=width, size_x, size_y, T + padding]
        
        x = x1 + x2
        x = nn.functional.gelu(x)
        # input: [batchsize, out_channel=width, size_x, size_y, T + padding]
        # output: [batchsize, out_channel=width, size_x, size_y, T + padding]

        # self.memory.print("after FNO1")
        
        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = nn.functional.gelu(x)
        # self.memory.print("after FNO2")

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = nn.functional.gelu(x)
        # self.memory.print("after FNO3")

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        # self.memory.print("after FNO4")
        # output: [batchsize, out_channel=width, size_x, size_y, T + padding]
        
        x = x[..., :-self.padding]
        # output: [batchsize, out_channel=width, size_x, size_y, T]
        
        x = self.q(x) 
        # MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)
            # x = self.mlp1(x)
            # input: [batchsize, in_channel=width, size_x, size_y, T ]
            # weight: [mid_channel=4*width, in_channel=width, 1,1,1]
            # output: [batchsize, out_channel=mid_channel=4*width, size_x, size_y, T ]
            # x = torch.nn.Functional.gelu(x)
            # input: [batchsize, mid_channel=4*width, size_x, size_y, T ]
            # output: [batchsize, mid_channel=4*width, size_x, size_y, T]
            # x = self.mlp2(x)
            # input: [batchsize, in_channel=mid_channel=4*width, size_x, size_y, T]
            # weight: [out_channel=1, mid_channel=4*width, 1, 1, 1]
            # output: [batchsize, out_channel=1, size_x, size_y, T]
            
        # self.memory.print("after q(x)")
        
        x = x.permute(0, 2, 3, 4, 1) 
        # [batchsize, out_channel=1, size_x, size_y, T] ---> [batchsize, size_x, size_y, T, out_channel=1]
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)  # [batchsize, size_x, size_y, T, 3]
    
    def print_size(self):
        properties = []

        for param in self.parameters():
            properties.append([list(param.size()+(2,) if param.is_complex() else param.size()), param.numel(), (param.data.element_size() * param.numel())/1000])
            
        elementFrame = pd.DataFrame(properties, columns = ['ParamSize', 'NParams', 'Memory(KB)'])
 
        print(f'Total number of model parameters: {elementFrame["NParams"].sum()} with (~{format_tensor_size(elementFrame["Memory(KB)"].sum()*1000)})')
        return elementFrame

