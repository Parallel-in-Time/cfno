import h5py
import numpy as np

def save_inference(iteration:int,
                    ux:np.ndarray, vx_pred:np.ndarray,
                    uy:np.ndarray, vy_pred:np.ndarray,
                    b_in:np.ndarray, b_pred:np.ndarray,
                    p_in:np.ndarray, p_pred:np.ndarray, 
                    time_in:list, time_out:list,
                    file:str, calc_loss:bool,
                    xStep:int=1, yStep:int=1, tStep:int=1, 
                    vx:np.ndarray=None, vy:np.ndarray=None,
                    b_out:np.ndarray=None, p_out:np.ndarray=None):
    """
    Write inference result into a hdf5 file

    Args:
        iteration (int): index for inference output
        ux (np.ndarray): input velocity x-component 
        vx_pred (np.ndarray): FNO output velocity x-component
        uy (np.ndarray): input velocity y-component 
        vy_pred (np.ndarray): FNO output velocity y-component
        b_in (np.ndarray): input buoyancy
        b_pred (np.ndarray): FNO output buoyancy
        p_in (np.ndarray): input pressure
        p_pred (np.np.ndarray): FNO ouput pressure 
        time_in (list): input simulation times
        time_out (list): output simulation times
        file (str): hdf5 file to store inference result 
        calc_loss (bool): if inference loss is calculated 
        xStep (int, optional): x-grid slicing. Defaults to 1.
        yStep (int, optional): y-grid slicing. Defaults to 1.
        tStep (int, optional): timestep slicing. Defaults to 1.
        vx (np.ndarray, optional): ouput velocity x-component. Defaults to None.
        vy (np.ndarray, optional): output velocity y-component. Defaults to None.
        b_out (np.ndarray, optional): output buoyancy. Defaults to None.
        p_out (np.ndarray, optional): output pressure. Defaults to None.
    """
    
    # Storing inference result
    with h5py.File(file, "a") as data:
        for index_in in range(0, len(time_in), tStep):
            data[f'inference_{iteration}/scales/sim_timein_{index_in}'] = time_in[index_in]
            data[f'inference_{iteration}/tasks/input/velocity_{index_in}'] = np.stack([ux[::xStep,::yStep, index_in], uy[::xStep,::yStep, index_in]], axis=0)
            data[f'inference_{iteration}/tasks/input/buoyancy_{index_in}'] = b_in[::xStep, ::yStep, index_in]
            data[f'inference_{iteration}/tasks/input/pressure_{index_in}'] = p_in[::xStep, ::yStep, index_in]
        for index_out in range(0, len(time_out), tStep):
            data[f'inference_{iteration}/scales/sim_timeout_{index_out}']= time_out[index_out]
            if calc_loss and vx is not None:
                data[f'inference_{iteration}/tasks/output/velocity_{index_out}']= np.stack([vx[::xStep,::yStep, index_out], vy[::xStep,::yStep, index_out]], axis=0)
                data[f'inference_{iteration}/tasks/output/buoyancy_{index_out}'] = b_out[::xStep, ::yStep, index_out]
                data[f'inference_{iteration}/tasks/output/pressure_{index_out}']= p_out[::xStep, ::yStep, index_out]
            data[f'inference_{iteration}/tasks/model_output/velocity_{index_out}']= np.stack([vx_pred[::xStep,::yStep, index_out], vy_pred[::xStep,::yStep, index_out]], axis=0)
            data[f'inference_{iteration}/tasks/model_output/buoyancy_{index_out}']= b_pred[::xStep, ::yStep, index_out]
            data[f'inference_{iteration}/tasks/model_output/pressure_{index_out}']= p_pred[::xStep, ::yStep, index_out]
                        