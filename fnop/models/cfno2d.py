import math

import torch as th
import torch.nn as nn

from torch_dct import dct, idct


class CF2DConv(nn.Module):
    """2D Neural Convolution, FFT in X, DCT in Y (can force FFT in Y for comparison)"""

    USE_T_CACHE = False

    def __init__(self, kX, kY, dv, forceFFT=False):
        super().__init__()

        self.kX = kX
        self.kY = kY
        self.forceFFT = forceFFT

        self.R = nn.Parameter(
            th.rand(dv, dv, kX, kY, dtype=th.cfloat))

        if forceFFT:
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

    def __init__(self, kX, kY, dv, forceFFT=False):
        super().__init__()

        self.conv = CF2DConv(kX, kY, dv, forceFFT)
        self.sigma = nn.ReLU(inplace=True)
        self.W = Grid2DLinear(dv, dv)


    def forward(self, x):
        """ x[nBatch, dv, nX, nY] -> [nBatch, dv, nX, nY] """

        v = self.conv(x)    # 2D Convolution
        w = self.W(x)       # Linear operator

        v += w
        o = self.sigma(v)

        return o


class CFNO2D(nn.Module):

    def __init__(self, da, dv, du, kX=4, kY=4, nLayers=1, forceFFT=False):
        super().__init__()

        self.P = Grid2DLinear(da, dv)
        self.Q = Grid2DLinear(dv, du)
        self.layers = nn.ModuleList(
            [CF2DLayer(kX, kY, dv, forceFFT) for _ in range(nLayers)])
        # self.pos = Grid2DPartialPositiver([0, 0, 1, 1])


    def forward(self, x):
        """ x[nBatch, da, nX, nY] -> [nBatch, du, nX, nY]"""

        x = self.P(x)
        for layer in self.layers:
            x = layer(x)
        x = self.Q(x)
        # x = self.pos(x)

        return x


if __name__ == "__main__":
    # Quick script testing
    model = CFNO2D(4, 4, 4, nLayers=4, kX=12, kY=6)
    uIn = th.rand(5, 4, 256, 64)
    print(model(uIn).shape)
