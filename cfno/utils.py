import torch
import torch.nn.functional as F
import torch.distributed as dist
import signal
import math
import yaml
from configmypy import  Bunch


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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

def get_world_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
    return world_size

def get_device(local_rank=None):
    backend = torch.distributed.get_backend()
    if backend == 'nccl':
        if local_rank is None:
            device = torch.device('cuda')
        else:
            device = torch.device(f'cuda:{local_rank}')
    elif backend == 'gloo':
        device = torch.device('cpu')
    else:
        raise RuntimeError
    return device

def all_gather_item(item, dtype, group=None, async_op=False, local_rank=None):
    if not torch.distributed.is_available() or \
       not torch.distributed.is_initialized():
        return [item]

    device = get_device(local_rank)

    if group is not None:
        group_size = group.size()
    else:
        group_size = get_world_size()

    tensor = torch.tensor([item], device=device, dtype=dtype)
    output_tensors = [
        torch.zeros(1, dtype=tensor.dtype, device=tensor.device)
        for _ in range(group_size)
    ]
    torch.distributed.all_gather(output_tensors, tensor, group, async_op)
    output = [elem.item() for elem in output_tensors]
    return output
    
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

def print_rank0(message):
    """
    If distributed training is initiliazed, print only on rank 0
    """
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def add_channelwise_noise(
    input,
    noise_levels,  # (u, v, b, p)
):
    """
    input: (B, channels, nx, ny)
    returns noisy version of x with same shape
    """
    x_noisy = input.clone()

    for c, eps in enumerate(noise_levels):
        if eps == 0:
            continue

        # std per sample
        std = input[:, c].std(dim=(-2, -1), keepdim=True)  # (B,1,1)
        # Normal distribution with mean zero and eps*std scaled per batch
        noise = eps * std * torch.randn_like(input[:, c])  # (B,nx,ny)
        x_noisy[:, c] += noise

    return x_noisy

def augment_batch_with_noise(
    x,
    y,
    noise_levels,
):
    """
    Keeps all clean samples and appends noisy copies.

    Input:
        x : (B, channels, nx, ny)
        y : (B, channels, nx, ny)

    Output:
        x_aug : (2B, channels, nx, ny)
        y_aug : (2B, channels, nx, ny)
    """
    x_noisy = add_channelwise_noise(x, noise_levels)

    x_aug = torch.cat([x, x_noisy], dim=0)
    y_aug = torch.cat([y, y], dim=0)

    return x_aug, y_aug

def compile_timing(func):
    """
    Function to return timing in seconds
    and result of running func.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = func()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000