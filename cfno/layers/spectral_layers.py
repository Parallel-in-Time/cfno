import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    """
    2D Spectral convolution

    Args:
        in_channels (int): input channels
        out_channels (int): output channels
        modes1, modes2 (int): Fourier modes
  
    """
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int,
                 modes1:int, 
                 modes2:int
    ):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        # k_max = 12 in http://arxiv.org/pdf/2010.08895
        self.modes1 = modes1           
        self.modes2 = modes2

        # R
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        # summation along in_channel
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        # x = [batchsize, width, size_x+padding, size_y+padding]
        batchsize = x.shape[0]
        
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        # [batchsize, width, size_x+padding, if (size_y+padding) is even ((size_y+padding)/2 +1) else (size_y+padding)/2]
    
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1))) # x = [batchsize, width, size_x+padding, size_y+padding]
        return x

class SpectralConv3d(nn.Module):
    """
    3D Spectral convolution

    Args:
        in_channels (int): input channels
        out_channels (int): output channels
        modes1, modes2, modes3 (int): Fourier modes
  
    """
    def __init__(self,
                 in_channels:int,
                 out_channels:int, 
                 modes1:int,
                 modes2:int, 
                 modes3:int
    ):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels  
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        # k_max = 12 in http://arxiv.org/pdf/2010.08895
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
        
        # Multiply relevant Fourier modes (Corners of R) ---> R.FFT(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)  # upper right
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2) # upper left
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3) # lower right
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4) # lower left

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1))) # x = [batchsize, width, size_x, size_y, T + padding]
        return x
