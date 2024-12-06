import torch
import torch.nn.functional as F
import torch.distributed as dist
import signal
import math
import yaml
from configmypy import ConfigPipeline, YamlConfig, Bunch

_GLOBAL_SIGNAL_HANDLER = None
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def read_config(config):
    """
    Read a configuration file for a FNO model

    Parameters
    ----------
    config : str or configmypy.bunch.Bunch
        The path of the configuration file, or the already loaded config file.

    Returns
    -------
    configmypy.bunch.Bunch
        Configuration parameters.
    """
    if isinstance(config, Bunch):
        return config
    assert isinstance(config, str), "config parameter must be a string"
    pipe = ConfigPipeline([YamlConfig(config)])
    return pipe.read_conf()

def readConfig(config):
    """
    Safe read config based on yaml, that does not convert scalar into weird ruaml types ...
    """
    with open(config, "r") as f:
        conf = yaml.safe_load(f)
    return Bunch(conf)

units = {
    0: 'B',
    1: 'KiB',
    2: 'MiB',
    3: 'GiB',
    4: 'TiB'
}

def format_mem(x):
    """
    Takes integer 'x' in bytes and returns a number in [0, 1024) and
    the corresponding unit.

    """
    if abs(x) < 1024:
        return round(x, 2), 'B'

    scale = math.log2(abs(x)) // 10
    scaled_x = x / (1024 ** scale)
    unit = units[scale]

    if int(scaled_x) == scaled_x:
        return int(scaled_x), unit

    # rounding leads to 2 or fewer decimal places, as required
    return round(scaled_x, 2), unit

def format_tensor_size(x):
    val, unit = format_mem(x)
    return f'{val}{unit}'

def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)

def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)

def get_signal_handler():
    _ensure_var_is_initialized(_GLOBAL_SIGNAL_HANDLER, 'signal handler')
    return _GLOBAL_SIGNAL_HANDLER

def _set_signal_handler():
    global _GLOBAL_SIGNAL_HANDLER
    _ensure_var_is_not_initialized(_GLOBAL_SIGNAL_HANDLER, 'signal handler')
    _GLOBAL_SIGNAL_HANDLER = DistributedSignalHandler().__enter__()

class CudaMemoryDebugger():
    """
    Helper to track changes in CUDA memory.

    """
    DEVICE = 'cuda'
    LAST_MEM = 0
    ENABLED = True


    def __init__(self, print_mem):
        self.print_mem = print_mem
        if not CudaMemoryDebugger.ENABLED:
            return

        cur_mem = torch.cuda.memory_allocated(CudaMemoryDebugger.DEVICE)
        cur_mem_fmt, cur_mem_unit = format_mem(cur_mem)
        print(f'cuda allocated (initial): {cur_mem_fmt:.2f}{cur_mem_unit}')
        CudaMemoryDebugger.LAST_MEM = cur_mem

    def print(self,id_str=None):
        if not CudaMemoryDebugger.ENABLED:
            return

        desc = 'cuda allocated'

        if id_str is not None:
            desc += f' ({id_str})'

        desc += ':'

        cur_mem = torch.cuda.memory_allocated(CudaMemoryDebugger.DEVICE)
        cur_mem_fmt, cur_mem_unit = format_mem(cur_mem)

        diff = cur_mem - CudaMemoryDebugger.LAST_MEM
        if self.print_mem:
            if diff == 0:
                print(f'{desc} {cur_mem_fmt:.2f}{cur_mem_unit} (no change)')

            else:
                diff_fmt, diff_unit = format_mem(diff)
                print(f'{desc} {cur_mem_fmt:.2f}{cur_mem_unit}'
                      f' ({diff_fmt:+}{diff_unit})')

        CudaMemoryDebugger.LAST_MEM = cur_mem

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001, time_last=True):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T in 1D
        # x could be in shape of ntrain*w*l or ntrain*T*w*l or ntrain*w*l*T in 2D
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps
        self.time_last = time_last # if the time dimension is the last dim

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        # sample_idx is the spatial sampling mask
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if self.mean.ndim == sample_idx.ndim or self.time_last:
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if self.mean.ndim > sample_idx.ndim and not self.time_last:
                    std = self.std[...,sample_idx] + self.eps # T*batch*n
                    mean = self.mean[...,sample_idx]
        # x is in shape of batch*(spatial discretization size) or T*batch*(spatial discretization size)
        x = (x * std) + mean
        return x

    def to(self, device):
        if torch.is_tensor(self.mean):
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        else:
            self.mean = torch.from_numpy(self.mean).to(device)
            self.std = torch.from_numpy(self.std).to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

class DistributedSignalHandler:
    def __init__(self, sig=signal.SIGTERM):
        self.sig = sig

    def signals_received(self):
        all_received = all_gather_item(
            self._signal_received, dtype=torch.int32
        )
        return all_received

    def __enter__(self):
        self._signal_received = False
        self.released = False
        self.original_handler = signal.getsignal(self.sig)

        def handler(signum, frame):
            self._signal_received = True

        signal.signal(self.sig, handler)

        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        if self.released:
            return False

        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True

def activation_selection(choice):
    if choice in ['tanh', 'Tanh']:
        return F.tanh
    elif choice in ['relu', 'ReLU']:
        return F.relu
    elif choice in ['sigmoid', 'Sigmoid']:
        return F.sigmoid
    elif choice in ['celu', 'CeLU']:
        return F.celu
    elif choice in ['gelu', 'GeLU']:
        return F.gelu
    elif choice in ['mish']:
        return F.mish
    else:
        raise ValueError('Unknown activation function')

def print_rank0(message):
    """
    If distributed training is initiliazed, print only on rank 0
    """
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def format_complexTensor(weight):
    """
    Convert torch.cfloat to torch.float32
    for torch DDP with NCCL communication
  
    """
  
    if weight.dtype == torch.complex64:
        R = torch.view_as_real(weight)
    else:
        R  = weight
    return R


def deformat_complexTensor(weight):  
    """
    Convert torch.float32 to torch.cfloat
    for computation
  
    """

    if weight.dtype != torch.complex64:
        R = torch.view_as_complex(weight)
    else:
        R  = weight
    return R

