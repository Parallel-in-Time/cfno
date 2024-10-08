import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fnop.utils import CudaMemoryDebugger, format_tensor_size
from fnop.layers.mlp import ChannelMLP
from fnop.layers.spectral_layers import SpectralConv2d
from fnop.layers.normalization_layers import InstanceNorm

class FNO2D(nn.Module):
    """
    FNO2D model: 2D space convolution and recurrent in time

    Args:
        modes1, modes2 (int): fourier modes
        lifting_width (int): width of lifting layer
        width (int): width of fourier layer
        projection_width (int): width of projecting layer
        n_layers (int): number of Fourier layers to apply in sequence. Default is 4.
        T_in (int): number of input timesteps. Default is 1.
        T (int): number of output timesteps. Default is 1.
        padding (int): padding to make input periodic if non-periodic
        n_dim (int): convolution dimension. Default is 2.
        use_grid (bool): add grid data into channel. Default is False.
        activation : activation function used. Default is nn.functional.gelu()
        use_channel_mlp (bool): whether to use ChannelMLP layers after each FNO layer. Default is False.
        channel_mlp_dropout (float) : dropout parameter for self.channel_mlp. Default is 0.0.
        channel_mlp_expansion (float): expansion parameter for self.channel_mlp, by default 0.5
        
    """
    
    def __init__(self, 
                 modes1:int,
                 modes2:int,
                 lifting_width:int,
                 width:int, 
                 projection_width:int,
                 n_layers:int=4,
                 T_in:int=1,
                 T:int=1,
                 activation=nn.functional.gelu,
                 n_dim:int=2,
                 use_grid:bool=False,
                 padding=None,
                 use_channel_mlp:bool=False,
                 channel_mlp_expansion:float=0.5,
                 channel_mlp_dropout:float=0.0
    ):
        super().__init__()

        """
        The overall network. It contains n layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.p 
        2. n_layers of the integral operators u' = (W + K)(u)
        3. Project from the channel space to the output space by self.q
        
        input: if use_grid then solution of previous T_in timesteps + n_dim locations 
              ((u(1, x, y), ..., u(T_in, x, y), x, y) else
              solution of previous T_in timesteps ((u(1, x, y), ..., u(T_in, x, y))
        input shape: (batchsize, x=4*size_x//xStep, y=size_y/yStep, c=T_in+n_dim)
        output: the solution of the next timestep 
        output shape: (batchsize, x=4*size_x//xStep, y=size_y//yStep, c=1)
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.lifting_width = lifting_width
        self.width = width
        self.projection_width = projection_width
        self.n_layers = n_layers
        self.T_in = T_in
        self.T = T
        self.padding = 0 if padding is None else padding
        self.n_dim = n_dim
        self.use_grid = use_grid
        self.activation = activation
        self.use_channel_mlp = use_channel_mlp

        # input channel is T_in + n_dim: the solution of T_in timesteps + n_dim locations 
        # (u(t, x, y), ..., u(t+T_in, x, y), x, y) if use_grid else solution of T_in timesteps
        if self.use_grid:
            self.input_channels = self.T_in + self.n_dim
        else:
            self.input_channels = self.T_in
        
        self.out_channels = 1
        # x = (batchsize, x=size_x, y=size_y, c=input_channel)
        # lifting layer 
        # self.p = ChannelMLP(in_channels=self.input_channels,
        #                     out_channels=self.width,
        #                     hidden_channels=self.lifting_width,
        #                     n_layers=2,
        #                     non_linearity=self.activation)
        self.p = nn.Sequential(nn.Linear(self.input_channels, self.lifting_width),  # scaling: p layer
                               nn.ReLU(inplace=True),
                               nn.Linear(self.lifting_width, self.width))   

        
        self.conv_list = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(self.n_layers)]) # W
        self.spectral_list = nn.ModuleList([SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(self.n_layers)])  # k
        if self.use_channel_mlp:
            self.channel_mlp = ChannelMLP(
                        in_channels=self.width,
                        out_channels=self.width,
                        hidden_channels=round(self.width * channel_mlp_expansion),
                        n_layers=2,
                        non_linearity=self.activation,
                        dropout=channel_mlp_dropout,
                    )
        else:
            self.channel_mlp = None
        
        # projection: q layer 
        # self.q = ChannelMLP(in_channels=self.width,
        #                     out_channels=self.out_channels,
        #                     hidden_channels=self.projection_width,
        #                     n_layers=2,
        #                     non_linearity=self.activation)  
        self.q = nn.Sequential(nn.Linear(self.width, self.projection_width),  # scaling: p layer
                               nn.ReLU(inplace=True),
                               nn.Linear(self.projection_width, self.out_channels))   
        
        self.norm = InstanceNorm()
        # self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.mlp0 = MLP(self.width, self.width, self.width, self.n_dim)
        # self.mlp1 = MLP(self.width, self.width, self.width, self.n_dim)
        # self.mlp2 = MLP(self.width, self.width, self.width, self.n_dim)
        # self.mlp3 = MLP(self.width, self.width, self.width, self.n_dim)
        # self.w0 = nn.Conv2d(self.width, self.width, 1)
        # self.w1 = nn.Conv2d(self.width, self.width, 1)
        # self.w2 = nn.Conv2d(self.width, self.width, 1)
        # self.w3 = nn.Conv2d(self.width, self.width, 1)
        # self.norm = nn.InstanceNorm2d(self.width)
        # self.q = MLP(self.width, 1, self.width * 4, self.n_dim) 
        
        self.memory = CudaMemoryDebugger(print_mem=True)

    def forward(self, x):
        
        if self.use_grid:
            grid = self.get_grid(x.shape, x.device) 
            x = torch.cat((x, grid), dim=-1)
            # [batchsize, size_x, size_y, c=T_in] ---> [batchsize, size_x, size_y, c=T_in+n_dim ]
            
        # x = x.permute(0, 3, 1, 2)  
        # print(f'x before p(x): {x.shape}')
        x = self.p(x)
        # print(f'x after p(x): {x.shape}')
        x = x.permute(0, 3, 1, 2)  
        # [batchsize, size_x, size_y, c=width] ---> [batchsize, width, size_x, size_y]
        
        # self.memory.print("after p(x)")
        
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        # padding order:(padding_left,padding_right, 
        #                 padding_top,padding_bottom,
        #                 padding_front,padding_back)
        if self.padding > 0:
            x = nn.functional.pad(x, [0, self.padding, 0, self.padding]) # pad the (x,y) domain if input is non-periodic
        # [batchsize, width, size_x+padding, size_y+padding]
        
        for index, (k, w) in enumerate(zip(self.spectral_list, self.conv_list)):

            x1 = k(x)
            x1 = self.norm(x1)
            x2 = w(x)
            x = x1 + x2
            if index != self.n_layers - 1:
                x = self.activation(x)
            if self.use_channel_mlp:
                x = self.channel_mlp(x)
                x = self.norm(x)
                x = self.activation(x)
            # self.memory.print(f"after FNO{index}")


        del x1
        del x2

        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding] # unpad the domain 
            # [batchsize, out_channel=width, size_x+padding, size_y+padding] ---> [batchsize, out_channel=width, size_x, size_y]

        x = x.permute(0, 2, 3, 1)
        x = self.q(x)
        # self.memory.print("after q(x)")
        
        # x = x.permute(0, 2, 3, 1)
        # [batchsize, out_channel=1, size_x, size_y] ---> [batchsize, size_x, size_y, out_channel=1]
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device) # [batchsize, size_x, size_y, 2]
    
    def print_size(self):
        properties = []

        for param in self.parameters():
            properties.append([list(param.size()+(2,) if param.is_complex() else param.size()), param.numel(), (param.data.element_size() * param.numel())/1000])
            
        elementFrame = pd.DataFrame(properties, columns = ['ParamSize', 'NParams', 'Memory(KB)'])
        total_param = elementFrame["NParams"].sum()
        total_mem = elementFrame["Memory(KB)"].sum()
        totals = pd.DataFrame(data=[[0, total_param, total_mem]], columns=['ParamSize', 'NParams', 'Memory(KB)'])
        elementFrame = pd.concat([elementFrame,totals], ignore_index=True, sort=False)
        print(f'Total number of model parameters: {total_param} with (~{format_tensor_size(total_mem*1000)})')
        return elementFrame


