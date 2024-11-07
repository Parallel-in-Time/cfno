import h5py
import numpy as np


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
                  nx:int,
                  ny:int,
                  t:int
):
    """
    Unpack stack [velx, velz, buoyancy, pressure]

    Args:
        result (np.ndarray): stack [velx, velz, buoyancy, pressure]
        nx (int): x size
        ny (int): y size
        t (int): timesteps to extract data for 

    Returns:
        ux (np.ndarray): velocity x-component
        uy (np.ndarray): velocity y-component
        b (np.ndarray): buoyancy
        p (np.ndarray): pressure 
    """
    ux = result[:nx, :ny, :t]
    uy = result[nx:2*nx, :ny, :t]
    b = result[2*nx:3*nx, :ny,:t]
    p = result[3*nx:, :ny,:t]
    # [nx, ny, time]
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

def concat_variables(self,data_path=None,
                 xStep:int=1, zStep:int=1, tStep:int=1,
                 start_iteration:int=0, end_iteration:int=100):
    index = 0
    inputs = []
    if data_path is not None:
        filename = f'{data_path}/input_data.h5'
        with h5py.File(filename, "w") as data:
            for i,file in enumerate(self.files):
                total_iterations = self.times(i).shape[0]
                print(f"index: {i}, file: {file}, total_iterations: {total_iterations}")
                print(f"Extracting iterations {start_iteration} to {end_iteration}....")
                for t in range(start_iteration, end_iteration+1, tStep):
                    vel_t, b_t, p_t = self.rbc_data(file, t, True, False)
                    inputs.append(np.concatenate((vel_t[0,::xStep,::zStep],
                                                  vel_t[1,::xStep,::zStep], b_t, p_t), axis = 0))
                    index = index + 1
            data['input'] = inputs
    else:
        for i,file in enumerate(self.files):
            total_iterations = self.times(i).shape[0]
            print(f"index: {i}, file: {file}, total_iterations: {total_iterations}")
            print(f"Extracting iterations {start_iteration} to {end_iteration}....")
            for t in range(start_iteration, end_iteration+1, tStep):
                vel_t, b_t, p_t = self.rbc_data(file, t, True, False)
                inputs.append(np.concatenate((vel_t[0,::xStep,::zStep],
                                              vel_t[1,::xStep,::zStep], b_t, p_t), axis = 0))
                index = index + 1

    return np.array(inputs)

def check_subdomain(grid:int, ndomain:int):
    """
    Function to check if sub-domain division 
    is compatible 

    Args:
        grid (int): 1-D size of grid 
        ndomain (int): number of subdomains

    Returns:
        sub_domain: size of sub-domain
    """
    if grid % ndomain == 0:
        sub_domain = grid//ndomain
        return sub_domain
    else:
        raise ValueError(f'{grid} is not divisible by {ndomain}')