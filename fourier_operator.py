import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Func

from utils import format_tensor_size, activation_selection

torch.manual_seed(0)
np.random.seed(0)

class FourierFeatures(nn.Module):

    def __init__(self, scale, mapping_size, device):
        super().__init__()
        self.mapping_size = mapping_size
        self.scale = scale
        self.B = scale * torch.randn((self.mapping_size, 2)).to(device)  # [mapping_size,2]

    def forward(self, x):
        # x is the set of coordinate and it is passed as a tensor (nx, ny, 2)
        if self.scale != 0:
            x_proj = torch.matmul((2. * np.pi * x), self.B.T) # [nx, ny, 2]*[2,mapping_size] ----> [nx, ny, mapping_size]
            inp = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1) # [nx, ny, 2*mapping_size]
            return inp
        else:
            return x
        
        
################################################################
# 1D Fourier Layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. Performs FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights_1d = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def complexmulti_1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0] # x.shape == [batch_size, in_channels, number of grid points]
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        # real FFT is redundant along that last dimension.
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.complexmulti_1d(x_ft[:, :, :self.modes1], self.weights_1d)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, fno_architecture, in_channels = 2, out_channels=1, padding_frac= 1/4, device="cpu"):
        super(FNO1d, self).__init__()

        """
        The network contains n_layers of the Fourier layer. 
        The following is done:
        1. Lift the input to the desire channel dimension by self.fc0 (p layer)
        2. n layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv 
        3. Project from the channel space to the ouput space by self.fc1 (q layer)
        4. Ouput layer given by self.fc2

        input: solution of the initial condition and location (a(x), x)
        input_shape: (batchsize, x=s, c=2)
        output: solution at a later timestep
        output_shape: (batchsize, x=s, c=1)
        """

        self.padding_frac = padding_frac
        self.modes1 = fno_architecture["modes"]
        self.width = fno_architecture["width"]
        self.n_layers = fno_architecture["n_layers"]
        self.proj_scale = fno_architecture["proj_scale"]
        self.device = device
        self.input_dim = in_channels
        
        #lifting the input to fc0 (P layer)
        self.fc0 = nn.Linear(input_dim, self.width)  # input channel is 2: (u0(x), x)

        self.conv_list = nn.ModuleList(
            [nn.Conv1d(self.width, self.width, 1) for _ in range(self.n_layers)])  # K 
        self.spectral_list = nn.ModuleList(
            [SpectralConv1d(self.width, self.width, self.modes1) for _ in range(self.n_layers)]) # W
        
        self.fc1 = nn.Linear(self.width, self.proj_scale)  # projecting from fc0 --> fc1 (Q layer)
        self.fc2 = nn.Linear(self.proj_scale, 1)  # output layer 

        self.to(device)

    def forward(self, x):
     
        x = self.fc0(x)   # Lifting: P layer
        x = x.permute(0, 2, 1)  # (batch_size, nfun+1, width) ---> (batch_size, width, nfun+1)
        x_padding = int(round(x.shape[-1] * self.padding_frac))
        x = Func.pad(x, [0, x_padding])  # pad the domain if input is non-periodic
    
        for k, (s, c) in enumerate(zip(self.spectral_list, self.conv_list)):

            x1 = s(x)
            x2 = c(x)
            x = x1 + x2
            if k != self.n_layers - 1:
                x = Func.gelu(x)

        x = x[..., :-x_padding]
        x = x.permute(0, 2, 1) # (batch_size, width, nfun+1) ---> (batch_size, nfun+1, width)
        x = self.fc1(x)  # Projecting: Q layer
        x = Func.gelu(x)
        x = self.fc2(x)  # output layer
        return x.squeeze(-1)
    
     def print_size(self):
        properties = []

        for param in self.parameters():
            properties.append([list(param.size()+(2,) if param.is_complex() else param.size()), param.numel(), (param.data.element_size() * param.numel())/1000])
            
        elementFrame = pd.DataFrame(properties, columns = ['ParamSize', 'NParams', 'Memory(KB)'])
 
        print(f'Total number of model parameters: {elementFrame["NParams"].sum()} with (~{format_tensor_size(elementFrame["Memory(KB)"].sum()*1000)})')
        return elementFrame
    
    
    
################################################################
# 2D Fourier Layer
################################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. Performs FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def complexmulti_2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]  # (batch, in_channel, x,y )
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        # real FFT is redundant along that last dimension.
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.complexmulti_2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.complexmulti_2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, fno_architecture, in_channels=1, out_channels=1, device="cpu"):
        super(FNO2d, self).__init__()

        """
        The network contains n_layers of the Fourier layer. 
        The following is done:
        1. Lift the input layer to the desire channel dimension by p layer
        2. n layers of the integral operators u' = (W + K)(u).
           W defined by self.w; K defined by self.conv 
        3. Project from the channel space to the output space by q 
        

        input: solution of the initial condition and location (a(x,y), x, y)
        input_shape: (batchsize, x=sx, y=sy, c=3)
        output: solution at a later timestep
        output_shape: (batchsize, x=sx, y=sy, c=1)
        """
        self.modes1 = fno_architecture["modes1"]
        self.modes2 = fno_architecture["modes2"]
        self.width = fno_architecture["width"]
        self.n_layers = fno_architecture["n_layers"]
        self.proj_scale = fno_architecture["proj_scale"]
        self.padding = fno_architecture["padding"]
        self.include_grid = fno_architecture["include_grid"]
        self.input_dim = in_channels
        self.activation  = activation_selection(fno_architecture["activation"])
        self.device = device
        
      
    
        self.p = nn.Sequential(nn.Linear(self.input_dim, self.proj_scale),  # scaling: p layer
                                   self.activation,
                                   nn.Linear(self.proj_scale, self.width))   
        
     
        self.conv_list = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(self.n_layers)]) # K
        self.spectral_list = nn.ModuleList([SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(self.n_layers)])  # W

        
        self.q = nn.Sequential(nn.Linear(self.width, self.proj_scale), # projection: q layer 
                                self.activation,
                                nn.Linear(self.proj_scale, out_channels))  
        
        self.to(self.device)
                

    def forward(self, x):
 
        x = self.p(x)
        x = x.permute(0, 3, 1, 2) # (batch_size, x, y, width) ---> (batch_size, width, x, y)
        
        x1_padding =  self.padding
        x2_padding =  self.padding
                
        if self.padding>0: 
            x = Func.pad(x, [0, x1_padding, 0, x2_padding])

        for k, (s, c) in enumerate(zip(self.spectral_list, self.conv_list)):

            x1 = s(x)
            x2 = c(x)
            x = x1 + x2
            if k != self.n_layers - 1:
                x = self.activation(x)
        
        del x1
        del x2
        
        if self.padding > 0:
            x = x[..., :-x1_padding, :-x2_padding]            
        x = x.permute(0, 2, 3, 1)  # (batch_size, width, x, y) ---> (batch_size, x, y, width) 
        x = self.q(x)

        return x

    def print_size(self):
        properties = []

        for param in self.parameters():
            properties.append([list(param.size()+(2,) if param.is_complex() else param.size()), param.numel(), (param.data.element_size() * param.numel())/1000])
            
        elementFrame = pd.DataFrame(properties, columns = ['ParamSize', 'NParams', 'Memory(KB)'])
 
        print(f'Total number of model parameters: {elementFrame["NParams"].sum()} with (~{format_tensor_size(elementFrame["Memory(KB)"].sum()*1000)})')
        return elementFrame
    