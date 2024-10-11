import torch as th
import torch.nn as nn

from torch_dct import dct, idct


class CF2DConv(nn.Module):
    """2D Neural Convolution, FFT in X, DCT in Y"""

    def __init__(self, kX, kY, dv):
        super().__init__()

        self.kX = kX
        self.kY = kY

        # Layer's parameters : kX * kY * dv * dv
        self.R = nn.Parameter(th.rand(kX, kY, dv, dv, dtype=th.cfloat))


    def T(self, kMax, n):
        return th.cat([th.eye(kMax, dtype=th.cfloat), th.zeros(kMax, n-kMax)], dim=1)


    def forward(self, x:th.tensor):
        """ x[nBatch, nX, nY, dv] -> x[nBatch, nX, nY, dv] """

        # Permute dimensions -> [nBatch, dv, nX, nY]
        x = x.movedim(-1, -3)

        # DCT on last dimension -> [nBatch, dv, nX, nY]
        x = dct(x, norm="ortho")

        # RFFT on before-last dimension -> [nBatch, dv, nX, nY]
        x = th.fft.rfft(x, dim=-2, norm="ortho")

        # Truncate and keep only first modes -> [nBatch, dv, kX, kY]
        nX, nY = x.shape[-2:]
        Tx, Ty = self.T(self.kX, nX), self.T(self.kY, nY)
        x = th.einsum("ax,by,eixy->eiab", Tx, Ty, x)

        # Apply R -> [nBatch, dv, kX, kY]
        x = th.einsum("ijab,ejab->eiab", self.R, x)

        # Padding on high frequency modes -> [nBatch, dv, nX, nY]
        x = th.einsum("xa,yb,eiab->eixy", Tx.T, Ty.T, x)

        # IRFFT on before-last dimension -> [nBatch, dv, nX, nY]
        x = th.fft.irfft(x, dim=-2, norm="ortho")

        # IDCT on last dimension -> [nBatch, dv, nX, nY]
        x = idct(x, norm="ortho")

        # Permute dimensions -> [nBatch, nX, nY, dv]
        x = x.movedim(-3, -1)

        return x


class CF2DLayer(nn.Module):

    def __init__(self, kX, kY, dv):
        super().__init__()

        self.conv = CF2DConv(kX, kY, dv)
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

    def __init__(self, da, dv, du, kX=4, kY=4, nLayers=1):
        super().__init__()

        self.P = nn.Linear(da, dv)
        self.Q = nn.Linear(dv, du)
        self.layers = [CF2DLayer(kX, kY, dv) for _ in range(nLayers)]


    def forward(self, x):
        """ x[nBatch, nX, nY, da] -> x[nBatch, nX, nY, du]"""

        x = self.P(x)
        for layer in self.layers:
            x = layer(x)
        x = self.Q(x)

        return x
