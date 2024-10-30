import math
import numpy as np

import torch as th
import torch.nn as nn
import pandas as pd
from torch_dct import dct, idct
from fnop.utils import CudaMemoryDebugger, format_tensor_size, activation_selection

class CF2DConv(nn.Module):
    """2D Neural Convolution, FFT in X, DCT in Y (can force FFT in Y for comparison)"""

    USE_T_CACHE = False

    def __init__(self, kX, kY, dv, forceFFT=False, reorder=False):
        super().__init__()

        self.kX = kX
        self.kY = kY
        self.forceFFT = forceFFT
        self.reorder = reorder

        self.R = nn.Parameter(
            th.rand(dv, dv, kX, kY, dtype=th.cfloat))

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

    def T(self, kMax, n, device):
        return th.cat([
            th.eye(kMax, dtype=th.cfloat, device=device),
            th.zeros(kMax, n-kMax, device=device)], dim=1)

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
        Tx, Ty = self.T(self.kX, fX, x.device), self.T(self.kY, fY, x.device)
        # -- Tx[kX, fX], Ty[kY, fY]
        x = th.einsum("ax,by,eixy->eiab", Tx, Ty, x)

        # Apply R[dv, dv, kX, kY] -> [nBatch, dv, kX, kY]
        x = th.einsum("ijab,ejab->eiab", self.R, x)

        # Padding on high frequency modes -> [nBatch, dv, fX, fY]
        x = th.einsum("xa,yb,eiab->eixy", Tx.T, Ty.T, x)

        # Transform back to Real space -> [nBatch, dv, nX, nY]
        x = self._toRealSpace(x)
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

    def __init__(self, kX, kY, dv, forceFFT=False, non_linearity='gelu', bias=True):
        super().__init__()

        self.conv = CF2DConv(kX, kY, dv, forceFFT)
        if non_linearity == 'gelu':
            self.sigma = nn.functional.gelu
        else:
            self.sigma = nn.ReLU(inplace=True)
        self.W = Grid2DLinear(dv, dv, bias)


    def forward(self, x):
        """ x[nBatch, dv, nX, nY] -> [nBatch, dv, nX, nY] """

        v = self.conv(x)    # 2D Convolution
        w = self.W(x)       # Linear operator

        v += w
        o = self.sigma(v)
        return o


class CFNO2D(nn.Module):

    def __init__(self, da, dv, du, kX=4, kY=4, nLayers=1, forceFFT=False, non_linearity='gelu', bias=True):
        super().__init__()
        self.config = {
            key: val for key, val in locals().items()
            if key != "self" and not key.startswith('__')}

        self.P = Grid2DLinear(da, dv, bias)
        self.Q = Grid2DLinear(dv, du, bias)
        self.layers = nn.ModuleList(
            [CF2DLayer(kX, kY, dv, forceFFT, non_linearity, bias)
             for _ in range(nLayers)])
        # self.pos = Grid2DPartialPositiver([0, 0, 1, 1])

        self.memory = CudaMemoryDebugger(print_mem=True)

    def forward(self, x):
        """ x[nBatch, nX, nY, da] -> [nBatch, du, nX, nY]"""
        # x = x.permute(0,3,1,2)
        x = self.P(x)

        for layer in self.layers:
            x = layer(x)

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
