import math
import numpy as np

import torch as th
import torch.nn as nn
import pandas as pd
from torch_dct import dct, idct
import torch.nn.functional as F

from cfno.layers.skip_connection import skip_connection
from cfno.utils import CudaMemoryDebugger, format_tensor_size, activation_selection

class CF2DConv(nn.Module):
    """2D Neural Convolution, FFT in X, DCT in Y (can force FFT in Y for comparison)"""

    USE_T_CACHE = False

    def __init__(self, kX, kY, dv, forceFFT=False, reorder=False, bias=False, order=2):
        super().__init__()

        self.kX = kX
        self.kY = kY
        self.forceFFT = forceFFT
        self.reorder = reorder
        self.order = order

        self.R = nn.Parameter(
            th.rand(dv, dv, kX*(2 if forceFFT else 1), kY, dtype=th.cfloat))
        
        if bias:
            self.init_std = (2 / (dv + dv))**0.5
            self.bias = nn.Parameter(
                self.init_std * th.randn(*(tuple([dv]) + (1,) * self.order))
            )
        else:
            self.init_std = None
            self.bias = None

        if forceFFT:
            if reorder:
                self._toFourierSpace = self._toFourierSpace_FORCE_FFT_REORDER
                self._toRealSpace = self._toRealSpace_FORCE_FFT_REORDER
            else:
                self._toFourierSpace = self._toFourierSpace_FORCE_FFT
                self._toRealSpace = self._toRealSpace_FORCE_FFT

        if self.USE_T_CACHE:
            self._T_CACHE = {}
            self.T = self.T_WITH_CACHE

    def T(self, kMax, n, device, sym=False):
        T = th.cat([
            th.eye(kMax, dtype=th.cfloat, device=device),
            th.zeros(kMax, n-kMax, device=device)], 
            dim=1)
        if sym:
            Tinv = th.cat([
                th.zeros(kMax, n-kMax, device=device),
                th.eye(kMax, dtype=th.cfloat, device=device)],
                dim=1)
            T = th.cat([T, Tinv], dim=0)
        return T

    def T_WITH_CACHE(self, kMax, n, device):
        try:
            T = self._T_CACHE[(kMax, n)]
        except KeyError:
            T = th.cat([
                    th.eye(kMax, dtype=th.cfloat, device=device),
                    th.zeros(kMax, n-kMax, device=device)], dim=1)
            self._T_CACHE[(kMax, n)] = T
        return T

    def _toFourierSpace(self, x):
        """ x[nBatch, dv, nX, nY] -> [nBatch, dv, fX = nX/2+1, fY = nY] """
        x = dct(x, norm="ortho")                    # DCT on last dimension
        x = th.fft.rfft(x, dim=-2, norm="ortho")    # RFFT on before-last dimension
        return x

    def _toRealSpace(self, x):
        """ x[nBatch, dv, fX = nX/2+1, fY = nY] -> [nBatch, dv, nX, nY] """
        x = th.fft.irfft(x, dim=-2, norm="ortho")   # IRFFT on before-last dimension
        x = idct(x, norm="ortho")                   # IDCT on last dimension
        return x

    def _toFourierSpace_FORCE_FFT(self, x):
        """ x[nBatch, dv, nX, nY] -> [nBatch, dv, fX = nX/2+1, fY = nY/2+1] """
        x = th.fft.rfft2(x, norm="ortho")   # RFFT on last 2 dimensions
        return x

    def _toRealSpace_FORCE_FFT(self, x):
        """ x[nBatch, dv, fX = nX/2+1, fY = nY/2+1] -> [nBatch, dv, nX, nY]"""
        x = th.fft.irfft2(x, norm="ortho")  # IRFFT on last 2 dimensions
        return x

    def _toFourierSpace_FORCE_FFT_REORDER(self, x):
        """ x[nBatch, dv, nX, nY] -> [nBatch, dv, fX = nX/2+1, fY = nY/2+1] """
        nY = x.shape[-1]
        reorder = np.append(np.arange((nY+1)//2)*2, -np.arange(nY//2)*2 - 1 - nY % 2)
        x = x[:, :, :, reorder]
        x = th.fft.rfft2(x, norm="ortho")   # RFFT on last 2 dimensions
        return x

    def _toRealSpace_FORCE_FFT_REORDER(self, x):
        """ x[nBatch, dv, fX = nX/2+1, fY = nY/2+1] -> [nBatch, dv, nX, nY]"""
        x = th.fft.irfft2(x, norm="ortho")  # IRFFT on last 2 dimensions
        nY = x.shape[-1]
        reorder = np.zeros(nY, dtype=int)
        reorder[: nY - nY % 2 : 2] = np.arange(nY // 2)
        reorder[1::2] = nY - np.arange(nY // 2) - 1
        reorder[-1] = nY // 2
        x = x[:, :, :, reorder]
        return x


    def forward(self, x:th.tensor):
        """ x[nBatch, dv, nX, nY] -> [nBatch, dv, nX, nY] """
        # Transform to Fourier space -> [nBatch, dv, fX, fY]
        x = self._toFourierSpace(x)
        # Truncate and keep only first modes -> [nBatch, dv, kX, kY]
        fX, fY = x.shape[-2:]
        Tx, Ty = self.T(self.kX, fX, x.device, sym=self.forceFFT), self.T(self.kY, fY, x.device)
        # -- Tx[kX, fX], Ty[kY, fY]
        x = th.einsum("ax,by,eixy->eiab", Tx, Ty, x)

        # Apply R[dv, dv, kX, kY] -> [nBatch, dv, kX, kY]
        x = th.einsum("ijab,ejab->eiab", self.R, x)

        # Padding on high frequency modes -> [nBatch, dv, fX, fY]
        x = th.einsum("xa,yb,eiab->eixy", Tx.T, Ty.T, x)

        # Transform back to Real space -> [nBatch, dv, nX, nY]
        x = self._toRealSpace(x)

        if self.bias is not None:
            x = x + self.bias

        return x


class ChannelMLP(nn.Module):
    """ChannelMLP applies an arbitrary number of layers of 
    1d convolution and nonlinearity to the channels of input
    and is invariant to spatial resolution.

    Parameters
    ----------
    in_channels : int
    out_channels : int, default is None
        if None, same is in_channels
    hidden_channels : int, default is None
        if None, same is in_channels
    n_layers : int, default is 2
        number of linear layers in the MLP
    non_linearity : default is F.gelu
    dropout : float, default is 0
        if > 0, dropout probability
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        hidden_channels=None,
        n_layers=2,
        n_dim=2,
        non_linearity=F.gelu,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = (
            in_channels if hidden_channels is None else hidden_channels
        )
        self.non_linearity = non_linearity
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
            if dropout > 0.0
            else None
        )
        
        # we use nn.Conv1d for everything and roll data along the 1st data dim
        self.fcs = nn.ModuleList()
        for i in range(n_layers):
            if i == 0 and i == (n_layers - 1):
                self.fcs.append(nn.Conv1d(self.in_channels, self.out_channels, 1))
            elif i == 0:
                self.fcs.append(nn.Conv1d(self.in_channels, self.hidden_channels, 1))
            elif i == (n_layers - 1):
                self.fcs.append(nn.Conv1d(self.hidden_channels, self.out_channels, 1))
            else:
                self.fcs.append(nn.Conv1d(self.hidden_channels, self.hidden_channels, 1))

    def forward(self, x):
        reshaped = False
        size = list(x.shape)
        if x.ndim > 3:  
            # batch, channels, x1, x2... extra dims
            # .reshape() is preferable but .view()
            # cannot be called on non-contiguous tensors
            x = x.reshape((*size[:2], -1)) 
            reshaped = True

        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)

        # if x was an N-d tensor reshaped into 1d, undo the reshaping
        # same logic as above: .reshape() handles contiguous tensors as well
        if reshaped:
            x = x.reshape((size[0], self.out_channels, *size[2:]))

        return x


class Grid2DLinear(nn.Module):

    def __init__(self, inSize, outSize, bias=True):
        super().__init__()

        self.weights = nn.Parameter(th.empty((outSize, inSize)))
        if bias:
            self.bias = nn.Parameter(th.empty((outSize, 1, 1)))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters (same as in pytorch for nn.Linear)
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """ x[nBatch, inSize, nX, nY] -> [nBatch, outSize, nX, nY] """
        x = th.einsum("ij,ejxy->eixy", self.weights, x)
        if self.bias is not None:
            x += self.bias
        return x


class Grid2DPartialPositiver(nn.Module):

    def __init__(self, posIdx):
        super().__init__()

        posDiag = th.tensor(posIdx, dtype=bool)
        Tpos = th.diag(posDiag).to(th.get_default_dtype())
        Tneg = th.diag(~posDiag).to(th.get_default_dtype())
        self.register_buffer('Tpos', Tpos)
        self.register_buffer('Tneg', Tneg)
        self.sigma = nn.ReLU(inplace=True)

    def forward(self, x):
        xPos = th.einsum("ij,ejxy->eixy", self.Tpos, x)
        xNeg = th.einsum("ij,ejxy->eixy", self.Tneg, x)
        xPos = self.sigma(xPos)
        out = xNeg + xPos
        return out


class CF2DLayer(nn.Module):

    def __init__(self, kX, kY, dv, 
                 forceFFT=False, 
                 non_linearity='gelu',
                 bias=False, reorder=False,
                 use_fno_skip_connection=False, 
                 fno_skip_type='linear'
                 ):
        super().__init__()

        self.conv = CF2DConv(kX, kY, dv, forceFFT, reorder, bias)
        if non_linearity == 'gelu':
            self.sigma = nn.functional.gelu
        else:
            self.sigma = nn.ReLU(inplace=True)
        if use_fno_skip_connection:
            self.W = skip_connection(dv, dv, skip_type=fno_skip_type)
        else:
            self.W = Grid2DLinear(dv, dv, bias)


    def forward(self, x):
        """ x[nBatch, dv, nX, nY] -> [nBatch, dv, nX, nY] """

        v = self.conv(x)    # 2D Convolution
        w = self.W(x)       # Linear operator

        v += w
        o = self.sigma(v)
        return o


class CFNO2D(nn.Module):

    def __init__(self, da, dv, du, kX=4, kY=4, 
                 nLayers=1,
                 forceFFT=False, 
                 non_linearity='gelu',
                 bias=True, 
                 reorder=False, 
                 use_prechannel_mlp=False,
                 use_fno_skip_connection=False, 
                 fno_skip_type='linear',
                 use_postfnochannel_mlp=False,
                 channel_mlp_skip_type='soft-gating',
                 channel_mlp_expansion=4
                 ):
        
        super().__init__()
        self.config = {
            key: val for key, val in locals().items()
            if key != "self" and not key.startswith('__')}
        
        self.use_postfnochannel_mlp = use_postfnochannel_mlp
        
        if use_prechannel_mlp:
            self.P = ChannelMLP(
                in_channels=da,
                out_channels=dv,
                hidden_channels=dv*channel_mlp_expansion,
                n_layers=4
                )
            self.Q =  ChannelMLP(
                in_channels=dv,
                out_channels=du,
                hidden_channels=dv*channel_mlp_expansion,
                n_layers=4
                )
        else:
            self.P = Grid2DLinear(da, dv, bias)
            self.Q = Grid2DLinear(dv, du, bias)

        self.layers = nn.ModuleList(
            [CF2DLayer(kX, kY, dv, forceFFT, non_linearity, bias, reorder,
                       use_fno_skip_connection,
                       fno_skip_type)
             for _ in range(nLayers)])
        # self.pos = Grid2DPartialPositiver([0, 0, 1, 1])

        if self.use_postfnochannel_mlp:
            postchannel_mlp_expansion = 0.5
            self.channel_mlp = nn.ModuleList(
            [ChannelMLP(in_channels=dv,
                        hidden_channels=round(dv * postchannel_mlp_expansion))
                for _ in range(nLayers)])
            
            self.channel_mlp_skips = nn.ModuleList(
            [skip_connection(dv, dv, skip_type=channel_mlp_skip_type)
                for _ in range(nLayers)])

        self.memory = CudaMemoryDebugger(print_mem=True)

    def forward(self, x):
        """ x[nBatch, nX, nY, da] -> [nBatch, du, nX, nY]"""
        # x = x.permute(0,3,1,2)
        x = self.P(x)

        for index,layer in enumerate(self.layers):
            if self.use_postfnochannel_mlp:
                x_skip_channel_mlp = self.channel_mlp_skips[index](x)

            x = layer(x)

            if self.use_postfnochannel_mlp:
                 x = self.channel_mlp[index](x) + x_skip_channel_mlp
                 if index < len(self.layers) - 1:
                    x = nn.functional.gelu(x)

        x = self.Q(x)
        # x = self.pos(x)
        # x = x.permute(0,2,3,1)

        return x


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

if __name__ == "__main__":
    # Quick script testing
    model = CFNO2D(4, 4, 4, nLayers=4, kX=12, kY=6)
    uIn = th.rand(5, 4, 256, 64)
    print(model(uIn).shape)
