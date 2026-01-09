from typing import Optional, Tuple, Union
import torch
from torch import nn

import tensorly as tl
from tensorly.plugins import use_opt_einsum
from neuralop.layers.base_spectral_conv import BaseSpectralConv
from neuralop.layers.resample import resample
from cfno.utils import format_complexTensor, deformat_complexTensor

Number = Union[int, float]

tl.set_backend("pytorch")
use_opt_einsum("optimal")
einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def _contract_dense(x, weight):
    order = tl.ndim(x)
    # batch-size, in_channels, x, y...
    x_syms = list(einsum_symbols[:order])

    # in_channels, out_channels, x, y...
    weight_syms = list(x_syms[1:])  # no batch-size

    # batch-size, out_channels, x, y...

    weight_syms.insert(1, einsum_symbols[order])  # outputs
    out_syms = list(weight_syms)
    out_syms[0] = x_syms[0]

    eq = f'{"".join(x_syms)},{"".join(weight_syms)}->{"".join(out_syms)}'

    if not torch.is_tensor(weight):
        weight = weight.to_tensor()

    return tl.einsum(eq, x, weight)


class SpectralConv(BaseSpectralConv):
    """SpectralConv implements the Spectral Convolution component of a Fourier layer
    described. 
    
    It is implemented as described in [1]_ and [2]_.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    n_modes : int or int tuple
        Number of modes to use for contraction in Fourier domain during training.

        .. warning::

            We take care of the redundancy in the Fourier modes, therefore, for an input
            of size I_1, ..., I_N, please provide modes M_K that are I_1 < M_K <= I_N
            We will automatically keep the right amount of modes: specifically, for the
            last mode only, if you specify M_N modes we will use M_N // 2 + 1 modes
            as the real FFT is redundant along that last dimension. For more information on
            mode truncation, refer to :ref:`fourier_layer_impl`


        .. note::

            Provided modes should be even integers. odd numbers will be rounded to the closest even number.

        This can be updated dynamically during training.
    max_n_modes : int tuple or None, optional
        * If not None, **maximum** number of modes to keep in Fourier Layer, along each dim
            The number of modes (`n_modes`) cannot be increased beyond that.
        * If None, all the n_modes are used.
        By default None.
    bias : bool, optional
        Whether to add a learnable bias to the output, by default True.
    fno_block_precision : str, optional
        Precision mode for FNO block operations. Options: 'full', 'half', 'mixed'.
        By default 'full'.
    rank : float, optional
        Rank of the tensor factorization of the Fourier weights, by default 1.0.
        Ignored if ``factorization is None``.
    factorization : str or None, optional
        Tensor factorization type. Options: {'tucker', 'cp', 'tt'}.
        If None, a single dense weight is learned for the FNO.
        Otherwise, that weight, used for the contraction in the Fourier domain
        is learned in factorized form. In that case, `factorization` is the
        tensor factorization of the parameters weight used.
        By default None.
    implementation : {'factorized', 'reconstructed'}, optional
        If factorization is not None, forward mode to use:
        * `reconstructed` : the full weight tensor is reconstructed from the
            factorization and used for the forward pass
        * `factorized` : the input is directly contracted with the factors of
            the decomposition
        Ignored if ``factorization is None``.
        By default 'reconstructed'.
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False.
        Ignored if ``factorization is None``.
    decomposition_kwargs : dict or None, optional
        Optional additional parameters to pass to the tensor decomposition.
        Ignored if ``factorization is None``.
        By default None.
    init_std : float or 'auto', optional
        Standard deviation to use for weight initialization, by default 'auto'.
        If 'auto', uses (2 / (in_channels + out_channels)) ** 0.5.
    fft_norm : str, optional
        FFT normalization parameter, by default 'forward'.
    device : torch.device or None, optional
        Device to place the layer on, by default None.
    

    References
    -----------
    .. [1] :

    Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential
        Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.

    .. [2] :

    Kossaifi, J., Kovachki, N., Azizzadenesheli, K., Anandkumar, A. "Multi-Grid
        Tensorized Fourier Neural Operator for High-Resolution PDEs" (2024).
        TMLR 2024, https://openreview.net/pdf?id=AWiDlO63bH.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        complex_data=False,
        max_n_modes=None,
        bias=True,
        separable=False,
        resolution_scaling_factor=None,
        fno_block_precision="full",
        rank=1.0,
        factorization=None,
        implementation="reconstructed",
        fixed_rank_modes=False,
        decomposition_kwargs: Optional[dict] = None,
        init_std="auto",
        fft_norm="forward",
        device=None,
    ):
        super().__init__(device=device)

        self.in_channels = in_channels
        self.out_channels = out_channels

        # n_modes is the total number of modes kept along each dimension
        self.n_modes = n_modes
        self.order = len(self.n_modes)
     
        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels)) ** 0.5

        self.fft_norm = fft_norm

        weight_shape = (in_channels, out_channels, *self.n_modes)
        # Create/init spectral weight tensor
        complex_weight = torch.rand(weight_shape, dtype=torch.cfloat)
        complex_weight.normal_(0, init_std)
        self.weight = nn.Parameter(format_complexTensor(complex_weight))

        self._contract = _contract_dense
        if bias:
            self.bias = nn.Parameter(
                init_std * torch.randn(*(tuple([self.out_channels]) + (1,) * self.order))
            )
        else:
            self.bias = None

    def transform(self, x, output_shape=None):
        in_shape = list(x.shape[2:])

        if output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            return resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        if isinstance(n_modes, int):  # Should happen for 1D FNO only
            n_modes = [n_modes]
        else:
            n_modes = list(n_modes)
        # the real FFT is skew-symmetric, so the last mode has a redundacy if our data is real in space
        # As a design choice we do the operation here to avoid users dealing with the +1
        # if we use the full FFT we cannot cut off informtion from the last mode
        n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes

    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None):
        """Generic forward pass for the Factorized Spectral Conv

        Parameters
        ----------
        x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        batchsize, channels, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        fft_size[-1] = fft_size[-1] // 2 + 1  # Redundant last coefficient in real spatial data
        fft_dims = list(range(-self.order, 0))


        x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)
        # When x is real in spatial domain, the last half of the last dim is redundant.
        # See :ref:`fft_shift_explanation` for discussion of the FFT shift.
        dims_to_fft_shift = fft_dims[:-1]

        if self.order > 1:
            x = torch.fft.fftshift(x, dim=dims_to_fft_shift)

        out_dtype = torch.cfloat
        out_fft = torch.zeros(
            [batchsize, self.out_channels, *fft_size], device=x.device, dtype=out_dtype
        )

        # if current modes are less than max, start indexing modes closer to the center of the weight tensor
        starts = [
            (max_modes - min(size, n_mode))
            for (size, n_mode, max_modes) in zip(fft_size, self.n_modes, self.n_modes)
        ]
       
        slices_w = [slice(None), slice(None)]  # in_channels, out_channels
        # The last mode already has redundant half removed in real FFT
        slices_w += [
            slice(start // 2, -start // 2) if start else slice(start, None)
            for start in starts[:-1]
        ]
        slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]

        slices_w = tuple(slices_w)
        weight = deformat_complexTensor(self.weight[slices_w].to(x.device))

        ### Pick the first n_modes modes of FFT signal along each dim
        # drop first two dims (in_channels, out_channels)
        weight_start_idx = 2
        slices_x = [slice(None), slice(None)]  # Batch_size, channels

        for all_modes, kept_modes in zip(fft_size, list(weight.shape[weight_start_idx:])):
            # After fft-shift, the 0th frequency is located at n // 2 in each direction
            # We select n_modes modes around the 0th frequency (kept at index n//2) by grabbing indices
            # n//2 - n_modes//2  to  n//2 + n_modes//2       if n_modes is even
            # n//2 - n_modes//2  to  n//2 + n_modes//2 + 1   if n_modes is odd
            center = all_modes // 2
            negative_freqs = kept_modes // 2
            positive_freqs = kept_modes // 2 + kept_modes % 2

            # this slice represents the desired indices along each dim
            slices_x += [slice(center - negative_freqs, center + positive_freqs)]

        if weight.shape[-1] < fft_size[-1]:
            slices_x[-1] = slice(None, weight.shape[-1])
        else:
            slices_x[-1] = slice(None)

        slices_x = tuple(slices_x)
        out_fft[slices_x] = self._contract(
            x[slices_x], weight
        )

        if output_shape is not None:
            mode_sizes = output_shape

        if self.order > 1:
            out_fft = torch.fft.ifftshift(out_fft, dim=fft_dims[:-1])

      
        x = torch.fft.irfftn(
            out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm
        )

        if self.bias is not None:
            x = x + self.bias

        return x