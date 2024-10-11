import torch as th
import torch.nn as nn

from torch_dct import dct, idct


class CF2DConv(nn.Module):
    """2D Neural Convolution, FFT in X, DCT in Y (can switch to FFT in Y for comparison)"""

    USE_T_CACHE = False

    def __init__(self, kX, kY, dv, forceFFT=False):
        super().__init__()

        self.kX = kX
        self.kY = kY
        self.forceFFT = forceFFT

        # Layer's parameters : kX * kY * dv * dv
        self.R = nn.Parameter(th.rand(kX, kY, dv, dv, dtype=th.cfloat))

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
        """ x[nBatch, dv, nX, nY] -> [nBatch, dv, nX, nY] """
        x = dct(x, norm="ortho")                    # DCT on last dimension
        x = th.fft.rfft(x, dim=-2, norm="ortho")    # RFFT on before-last dimension
        return x

    def _toRealSpace(self, x):
        """ x[nBatch, dv, nX, nY] -> [nBatch, dv, nX, nY] """
        x = th.fft.irfft(x, dim=-2, norm="ortho")   # IRFFT on before-last dimension
        x = idct(x, norm="ortho")                   # IDCT on last dimension
        return x

    def _toFourierSpace_FORCE_FFT(self, x):
        """ x[nBatch, dv, nX, nY] -> [nBatch, dv, nX, nY] """
        x = th.fft.rfft2(x, norm="ortho")   # RFFT on last 2 dimensions
        return x

    def _toRealSpace_FORCE_FFT(self, x):
        """ x[nBatch, dv, nX, nY] -> [nBatch, dv, nX, nY]"""
        x = th.fft.irfft2(x, norm="ortho")  # IRFFT on last 2 dimensions
        return x


    def forward(self, x:th.tensor):
        """ x[nBatch, nX, nY, dv] -> x[nBatch, nX, nY, dv] """

        # Permute dimensions -> [nBatch, dv, nX, nY]
        x = x.movedim(-1, -3)

        # Transform to Fourier space -> [nBatch, dv, nX, nY]
        x = self._toFourierSpace(x)

        # Truncate and keep only first modes -> [nBatch, dv, kX, kY]
        nX, nY = x.shape[-2:]
        Tx, Ty = self.T(self.kX, nX, x.device), self.T(self.kY, nY, x.device)
        x = th.einsum("ax,by,eixy->eiab", Tx, Ty, x)

        # Apply R -> [nBatch, dv, kX, kY]
        x = th.einsum("ijab,ejab->eiab", self.R, x)

        # Padding on high frequency modes -> [nBatch, dv, nX, nY]
        x = th.einsum("xa,yb,eiab->eixy", Tx.T, Ty.T, x)

        # Transform back to Real space -> [nBatch, dv, nX, nY]
        x = self._toRealSpace(x)

        # Permute dimensions -> [nBatch, nX, nY, dv]
        x = x.movedim(-3, -1)

        return x


class CF2DLayer(nn.Module):

    def __init__(self, kX, kY, dv, forceFFT=False):
        super().__init__()

        self.conv = CF2DConv(kX, kY, dv, forceFFT)
        self.sigma = nn.ReLU(inplace=True)
        self.W = nn.Linear(dv, dv)


    def forward(self, x):
        """ x[nBatch, nX, nY, dv] -> x[nBatch, nX, nY, dv] """

        v = self.conv(x)    # 2D Convolution
        w = self.W(x)       # Linear operator

        v += w
        o = self.sigma(v)   # Activation function

        return o


class CFNO2D(nn.Module):

    def __init__(self, da, dv, du, kX=4, kY=4, nLayers=1, forceFFT=False):
        super().__init__()

        self.P = nn.Linear(da, dv)
        self.Q = nn.Linear(dv, du)
        self.layers = nn.ModuleList(
            [CF2DLayer(kX, kY, dv, forceFFT) for _ in range(nLayers)])


    def forward(self, x):
        """ x[nBatch, nX, nY, da] -> x[nBatch, nX, nY, du]"""

        x = self.P(x)
        for layer in self.layers:
            x = layer(x)
        x = self.Q(x)

        return x
