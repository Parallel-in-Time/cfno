import h5py
import numpy as np
import torch

def rbc_data(filename:str,
             time:int,
             tasks=True, 
             scales=False
):
    """
    Extract data from dedalus hdf5 file

    Args:
        filename (str): path to dedauls hdf5 file
        time (int): time index
        tasks (bool, optional): extract states. Defaults to True.
        scales (bool, optional): extract scales. Defaults to False.

    Returns:
        out (tuple): extracted data
    """
    with h5py.File(filename, mode="r") as f:
        vel_t = f["tasks/velocity"][time]
        b_t = f["tasks/buoyancy"][time]
        p_t = f["tasks/pressure"][time]
        iteration = f["scales/iteration"][time]
        sim_time  = f["scales/sim_time"][time]
        time_step = f["scales/timestep"][time]
        wall_time = f["scales/wall_time"][time]
        write_no = f["scales/write_number"][time]

    out = tuple()
    if tasks:
        out += (vel_t, b_t, p_t)
    if scales:
        out += (write_no, iteration, sim_time, time_step, wall_time)
    if len(out) == 0:
        raise ValueError("Nothing to return!")
    return out

def state_extract(result:np.ndarray, 
                  gridx:int, 
                  gridy:int, 
                  t:int
):
    """
    Unpack stack [velx, velz, buoyancy, pressure]

    Args:
        result (np.ndarray): stack [velx, velz, buoyancy, pressure]
        gridx (int): x grid size
        gridy (int): y grid size
        t (int): timesteps to extract data for 

    Returns:
        ux (np.ndarray): velocity x-component
        uy (np.ndarray): velocity y-component
        b (np.ndarray): buoyancy
        p (np.ndarray): pressure 
    """
    ux = result[:gridx, :gridy, :t]
    uy = result[gridx:2*gridx, :gridy, :t]
    b = result[2*gridx:3*gridx, :gridy,:t]
    p = result[3*gridx:, :gridy,:t]
    
    # print(ux.shape, uy.shape, b.shape, p.shape)
    return ux, uy, b, p

def time_extract(time_index:int,
                 dt:float,
                 t_in:int=1,
                 t_out:int=1,
                 tStep:int=1
):
    """
    Extracting simulation time from dedalus data
    
    Args:
        time_index (int): time index
        dt (float): dedalus timestep 
        t_in (int, optional): number of input timesteps. Defaults to 1.
        t_out (int, optional): number of output timesteps. Defaults to 1.
        tStep (int, optional): time slices. Defaults to 1.

    Returns:
        time_in (list): list of simulation input times
        time_out (list): list of simulation output times
    """
    time_in = []
    for i in range(time_index, time_index + (t_in*tStep), tStep):
        time_in.append(i*dt)
    print("Input Time", time_in)
    time_out = []
    for j in range(time_index+(t_in*tStep), time_index + (t_in+t_out)*tStep, tStep):
        time_out.append(j*dt)
    print("Output Time", time_out)
    return time_in, time_out
    
def multi_data(reader, 
               task:str,
               start_time:int,
               end_time:int, 
               timestep:int,
               samples:int,
               T_in:int=1,
               T:int=1,
               xStep:int=1, 
               yStep:int=1, 
               tStep:int=1
):
    """
    Load data from multiple timesteps

    Args:
        reader : hdf5 reader
        task (str): 'train', 'val' or 'test'
        start_time (int): start time index
        end_time (int): end time index
        timestep (int): time interval 
        samples (int): number of simulations
        T_in (int, optional): number of input timesteps. Defaults to 1.
        T (int, optional): number of output timesteps. Defaults to 1.
        xStep (int, optional): slicing in x-grid. Defaults to 1.
        yStep (int, optional): slicing in y-grid. Defaults to 1.
        tStep (int, optional): time slicing. Defaults to 1.

    Returns:
        a_multi (torch.tensor): training input data 
        u_multi (torch.tensor): training output data
        
    """
    a = []
    u = []
    for index in range(start_time, end_time, timestep):
        a.append(torch.tensor(reader[task][:samples, ::xStep, ::yStep, index: index + (T_in*tStep): tStep], dtype=torch.float))
        u.append(torch.tensor(reader[task][:samples, ::xStep, ::yStep, index + (T_in*tStep): index + (T_in + T)*tStep: tStep], dtype=torch.float))
    a = torch.stack(a)
    u = torch.stack(u)
    
    a_multi = a.reshape(a.shape[0]*a.shape[1], a.shape[2], a.shape[3], a.shape[4])
    u_multi = u.reshape(u.shape[0]*u.shape[1], u.shape[2], u.shape[3], u.shape[4])
    
    return a_multi, u_multi