import os
import math
import torch
import torch.nn as nn


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

def print_tensor_mem(x, id_str=None):
    """
    Prints the memory required by tensor 'x'.

    """
    if not CudaMemoryDebugger.ENABLED:
        return

    desc = 'memory'

    if id_str is not None:
        desc += f' ({id_str})'

    desc += ':'

    val, unit = format_mem(x.element_size() * x.nelement())

    print(f'{desc} {val}{unit}')

def format_tensor_size(x):
    val, unit = format_mem(x)
    return f'{val}{unit}'


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


def activation_selection(choice):
    if choice in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif choice in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif choice in ['lrelu', 'LReLU']:
        return nn.LeakyReLU(inplace=True)
    elif choice in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif choice in ['softplus', 'Softplus']:
        return nn.Softplus(beta=4)
    elif choice in ['celu', 'CeLU']:
        return nn.CELU()
    elif choice in ['elu']:
        return nn.ELU()
    elif choice in ['mish']:
        return nn.Mish()
    else:
        raise ValueError('Unknown activation function')
    
    
def get_KESpectrum(field):

    u_x = field[0] # [Nx,Nz]
    u_z = field[1] # [Nx,Nz]
    eps = 1e-50 # to void log(0)

    # Compute the N-dimensional discrete Fourier Transform using FFT
    Ux_ampl = np.abs(np.fft.fftn(u_x)/u_x.size) # size = Nx*Nz
    Uz_ampl = np.abs(np.fft.fftn(u_z)/u_z.size) # size = Nx*Nz

    EK_Ux  = Ux_ampl**2
    EK_Uz  = Uz_ampl**2
    
    # Shift the zero-frequency component to the center of the spectrum.
    EK_Ux = np.fft.fftshift(EK_Ux)  # [Nx,Nz]
    EK_Uz = np.fft.fftshift(EK_Uz)  # [Nx,Nz]

    signal_sizex = np.shape(EK_Ux)[0] # [Nx]
    signal_sizey = np.shape(EK_Uz)[1] # [Nz]

    box_sidex = signal_sizex
    box_sidey = signal_sizey

    box_radius = int(np.ceil((np.sqrt((box_sidex)**2+(box_sidey)**2))/2.)+1)

    center_x = int(box_sidex/2)
    center_y = int(box_sidey/2)
    EK_Ux_avgsphr = np.zeros(box_radius,)+eps # size of the radius
    EK_Uz_avgsphr = np.zeros(box_radius,)+eps # size of the radius

    for i in range(box_sidex):
        for j in range(box_sidey):
            index =  int(np.round(np.sqrt((i-center_x)**2+(j-center_y)**2)))
            EK_Ux_avgsphr[index] = EK_Ux_avgsphr [index] + EK_Ux [i,j]
            EK_Uz_avgsphr[index] = EK_Uz_avgsphr [index] + EK_Uz [i,j]

    EK_avgsphr = 0.5*(EK_Ux_avgsphr + EK_Uz_avgsphr)

    return EK_avgsphr