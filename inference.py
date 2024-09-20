""""
Perform inference and plot results for 2D Rayleigh Benard Convection

Usage:
    python inference.py 
        --run=<run_tracker> 
        --model_checkpoint=<checkpoint> 
        --train_data_path=<train_data_path> (only when using FNO3D) 
        --test_data_path=<test_data_path>  
        --dim=FNO2D or FNO3D 
        --modes=12 
        --width=20 
        --batch_size=50 
        --rayleigh=1.5e7 
        --prandtl=1.0
        --gridx=256 
        --gridy=64 
        --train_samples=<train_samples> (only when using FNO3D) 
        --nTest=<test_samples> 
        --T_in=<input timesteps 
        --T=<output_timesteps> 
        --start_index=<dedalus_start_index> 
        --stop_index=<dedalus_stop_index> 
        --timestep=<dedalus_time_slice> 
        --dedalus_time_index=<absolute_dedalus_time_index>
        --dt=<dedalus_data_dt>
        --folder=<results>  
        --calc_loss
        
    optional args:
        --single_data_path<hdf5 file contains train, val and test data>
        --plotFile (only to plot without model_inference)
        --store_result 
        
"""

import os
import sys
import h5py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from timeit import default_timer
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from fno3d import FNO3d
from fno2d_recurrent import FNO2d
from utils import CudaMemoryDebugger, format_tensor_size, LpLoss, UnitGaussianNormalizer


def extract(result:np.ndarray, gridx:int, gridy:int, t:int):
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

def time_extract(time_index:int, dt:float, t_in:int=1, t_out:int=1, tStep:int=1):
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

def inferErrorPlot( ux:np.ndarray, vx:np.ndarray, vx_pred:np.ndarray,
                    uy:np.ndarray, vy:np.ndarray, vy_pred:np.ndarray,
                    b_in:np.ndarray, b_out:np.ndarray, b_pred:np.ndarray,
                    p_in:np.ndarray, p_out:np.ndarray, p_pred:np.ndarray, 
                    time_in:list, time_out:list, 
                    dim:str, fno_path:str,
                    gridx:int, gridy:int,
                    rayleigh:float, prandtl:float):
    """
    Plotting cross-sections of velocity, buoyancy and pressure data on grid
    with error plots 

    Args:
        ux (np.ndarray): Dedalus velocity x-component input 
        vx (np.ndarray): Dedalus velocity x-component output
        vx_pred (np.ndarray): FNO model velocty x-component output
        uy (np.ndarray): Dedalus velocity y-component input 
        vy (np.ndarray): Dedalus velocity y-component output
        vy_pred (np.ndarray): FNO model velocty y-component output
        b_in (np.ndarray): Dedalus buoyancy input
        b_out (np.ndarray): Dedalus buoyancy output
        b_pred (np.ndarray): FNO model buoyancy output
        p_in (np.ndarray):  Dedalus pressure input
        p_out (np.ndarray): Dedalus pressure output
        p_pred (np.ndarray): FNO model pressure output
        time_in (list): list of input simulation times
        time_out (list): list of output simulation times
        dim (str):  FNO2D or FNO3D strategy
        fno_path (str): path to store plots
        gridx (int): x grid size
        gridy (int): y grid size
        rayleigh (float): Rayleigh Number
        prandtl (float): Prandtl number 
    """
    for t in range(len(time_out)):
        row = 2
        col = 4
        xStep = 30
        yStep = 30
        x = np.arange(0,gridx,xStep)
        # fig = plt.figure(figsize=(16, 8))
        fig, ax = plt.subplots(nrows=row, ncols=col, figsize=(16, 12),
                               gridspec_kw={
                                   'width_ratios': [1,1,1,1],
                                   'height_ratios': [1,0.25],
                                   'wspace': 0.4,
                                   'hspace': 0.1})
        ax1 = ax[0][0]
        ax2 = ax[0][1]
        ax3 = ax[0][2]
        ax4 = ax[0][3]
        ax5 = ax[1][0]
        ax6 = ax[1][1]
        ax7 = ax[1][2]
        ax8 = ax[1][3]
   
        ax1.set_title(fr'Velocity: $u(x)$')
        # ax1.plot(x,ux[::xStep,::yStep,t],color='b',marker ='o',label="ux")
        ax1.plot(x,vx[::xStep,::yStep,t],color='g',marker ='o',label="ded-vx")
        ax1.plot(x,vx_pred[::xStep,::yStep,t],color='r',marker ='o',ls='--',label="fno-vx")
        # ax1.set_ylabel("Y grid")
        ax1.grid()

        ax2.set_title(fr'Velocity: $u(z)$ ')
        # ax2.plot(x,uy[::xStep,::yStep,t],color='b',marker ='o',label="uy")
        ax2.plot(x,vy[::xStep,::yStep,t],marker ='o',color='g',label="ded-vy")
        ax2.plot(x,vy_pred[::xStep,::yStep,t],marker ='o',color='r',linestyle='--',label="fno-vy")
        # ax2.set_ylabel("Y grid")
        ax2.grid()

        ax3.set_title(fr'Pressure: $p(x,z)$')
        # ax3.plot(x,p_in[::xStep,::yStep,t],color='b',marker ='o',label="p_in")
        ax3.plot(x,p_out[::xStep,::yStep,t],marker ='o',color='g',label="ded-p")
        ax3.plot(x,p_pred[::xStep,::yStep,t],marker ='o',color='r',linestyle='--',label="fno-p")
        # ax3.set_ylabel("Y grid")
        ax3.grid()

        ax4.set_title(fr'Buoyancy: $b(x,z)$')
        # ax4.plot(x,b_in[::xStep,::yStep,t],marker ='o',color='b',label="b_in")
        ax4.plot(x,b_out[::xStep,::yStep,t],marker ='o',color='g',label="ded-b")
        ax4.plot(x,b_pred[::xStep,::yStep,t],marker ='o',linestyle='--',color='r',label="fno-b")
        # ax4.set_ylabel("Y grid")
        ax4.grid()

        ax5.errorbar(x, np.average(vx[::xStep,::yStep,t],axis=1), yerr=np.average(np.abs(vx_pred[::xStep,::yStep,t]-vx[::xStep,::yStep,t]), axis=1), marker='o',color='purple',capsize=3, markersize=6,linestyle='none')
        ax5.set_ylabel(r"$\overline{|vx_{ded}-vx_{fno}|}_{z}$")
        ax5.set_xlabel("X Grid")

        ax6.errorbar(x, np.average(vy[::xStep,::yStep,t],axis=1), yerr=np.average(np.abs(vy_pred[::xStep,::yStep,t]-vy[::xStep,::yStep,t]), axis=1), marker='o',color='purple',capsize=3, markersize=6,linestyle='none')
        ax6.set_ylabel(r"$\overline{|vz_{ded}-vz_{fno}|}_{z}$")
        ax6.set_xlabel("X Grid")

        ax7.errorbar(x, np.average(p_out[::xStep,::yStep,t],axis=1), yerr=np.average(np.abs(p_pred[::xStep,::yStep,t]-p_out[::xStep,::yStep,t]), axis=1), marker='o',color='purple',capsize=3, markersize=6,linestyle='none')
        ax7.set_ylabel(r"$\overline{|p_{ded}-p_{fno}|}_{z}$")
        ax7.set_xlabel("X Grid")

        ax8.errorbar(x, np.average(b_out[::xStep,::yStep,t],axis=1), yerr=np.average(np.abs(b_pred[::xStep,::yStep,t]-b_out[::xStep,::yStep,t]), axis=1), marker='o',color='purple',capsize=3, markersize=6,linestyle='none')
        ax8.set_ylabel(r"$\overline{|b_{ded}-b_{fno}|}_{z}$")
        ax8.set_xlabel("X Grid")

        fig.suptitle(f'RBC-2D with {gridx}'+r'$\times$'+f'{gridy} grid and Ra={rayleigh} and Pr={prandtl} using {dim}')  
        if len(time_in) > 1:
            inp_patch = Line2D([0], [0], label=f'Input at t={np.round(time_in[0],4)}:{np.round(time_in[-1],4)}',marker='o', color='b')
        else:
            inp_patch = Line2D([0], [0], label=f'Input at t={np.round(time_in[0],4)}',marker='o', color='b')
        ded_patch = Line2D([0], [0], label=f'Dedalus at t={np.round(time_out[t],4)}',marker='o', color='g')
        fno_patch = Line2D([0], [0], label=f'FNO at t={np.round(time_out[t],4)}',marker='o', linestyle='--', color='r')
       
        fig.legend(handles=[inp_patch, ded_patch, fno_patch], loc="upper right")
        # fig.tight_layout()
        fig.show()
        fig.savefig(f"{fno_path}/{dim}_NX{gridx}_NY{gridy}_{np.round(time_out[t],4)}.png")

def data_loading(dim:str, test_data_path:str, batch_size:int,
                 gridx:int, gridy:int,
                 xStep:int=1, yStep:int=1, tStep:int=1,
                 T_in:int=1,T:int=1,
                 nTest:int=1, index:int=0, single_data_path:bool=False,
                 **kwargs):
    """
    Loading and pre-processing FNO input 

    Args:
        dim (str): FNO2D or FNO3D strategy
        test_data_path (str): path to hdf5 test data file
        batch_size (int): inference batch size 
        gridx (int): x grid size
        gridy (int): y grid size
        xStep (int, optional): x-grid slicing. Defaults to 1.
        yStep (int, optional): y-grid slicing. Defaults to 1.
        tStep (int, optional): timestep slicing. Defaults to 1.
        T_in (int, optional): number of input timesteps. Defaults to 1.
        T (int, optional): number of output timesteps. Defaults to 1.
        nTest (int, optional): number of test samples. Defaults to 1.
        index (int,optional): start index for T_in
        single_data_path (bool, optional): hdf5 file contains both test and/or train data
        train_data_path (str, optional): path to hdf5 train data file
        train_samples (int, optional): number of training samples
        
    Returns:
        test_loader (torch.DataLoader): (input, output) torch tensor for FNO 
    """
    train_data_path = ""
    train_samples = 0
    y_normalizer = None
    
    if 'train_samples' in kwargs.keys():
        train_samples = kwargs['train_samples']
    if 'train_data_path' in kwargs.keys():
        train_data_path = kwargs['train_data_path']
        
    if single_data_path:
        test_data_path = train_data_path  = test_data_path
        test_reader = train_reader = h5py.File(test_data_path, mode="r")
    else:
        if dim =='FNO3D':
            train_reader = h5py.File(train_data_path, mode="r")
        test_reader = h5py.File(test_data_path, mode="r")
    
    dataloader_time_start = default_timer()
    print('Starting data processing....')
    if dim == 'FNO3D':
        train_a = torch.tensor(train_reader['train'][:train_samples, ::xStep, ::yStep, index: index + (T_in*tStep): tStep], dtype=torch.float)
        train_u = torch.tensor(train_reader['train'][:train_samples, ::xStep, ::yStep, index + (T_in*tStep):  index + (T_in + T)*tStep: tStep], dtype=torch.float)
        # print(f"index: {index}")
        test_a = torch.tensor(test_reader['test'][:nTest, ::xStep, ::yStep, index: index + (T_in*tStep): tStep], dtype=torch.float)
        test_u = torch.tensor(test_reader['test'][:nTest, ::xStep, ::yStep, index + (T_in*tStep):  index + (T_in + T)*tStep: tStep], dtype=torch.float)
        dataloader_time_stop = default_timer()
        print(f'Total time taken for dataloading (s): {dataloader_time_stop - dataloader_time_start}')
        # Normalizing test data with train data
        a_normalizer = UnitGaussianNormalizer(train_a)
        test_a = a_normalizer.encode(test_a)
        y_normalizer = UnitGaussianNormalizer(train_u)
        test_u = y_normalizer.encode(test_u)
        # Repeating input for T timesteps
        test_a = test_a.reshape(nTest, gridx, gridy, 1, T_in).repeat([1,1,1,T,1])
        print(f'Total time take for data processing (s): {default_timer() - dataloader_time_start}')
        print(f"Test input data:{test_a.shape}, Test output data: {test_u.shape}")
    else:
        test_a = torch.tensor(test_reader['test'][:nTest, ::xStep, ::yStep, index: index + (T_in*tStep): tStep], dtype=torch.float)
        test_u = torch.tensor(test_reader['test'][:nTest, ::xStep, ::yStep, index + (T_in*tStep):  index + (T_in + T)*tStep: tStep], dtype=torch.float)
        dataloader_time_stop = default_timer()
        print(f'Total time taken for dataloading (s): {dataloader_time_stop - dataloader_time_start}')
        print(f"Test input data:{test_a.shape}, Test output data: {test_u.shape}")
    
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)
    if y_normalizer:
        return test_loader, y_normalizer
    else:
        return test_loader

def crosssection_plots(infFile:str, gridx:int, gridy:int,
                       dim:str,fno_path:str,rayleigh:float, prandtl:float,
                       T_in:int=1, T:int=1):
    """
    Plot cross-section plots 

    Args:
        infFile (str): path to inference hdf5 file
        gridx (int): x grid size
        gridy (int): y grid size
        dim (str): FNO2D or FNO3D strategy
        fno_path (str): path to store plots
        rayleigh (float): Rayleigh Number
        prandtl (float): Prandtl number 
        T_in (int, optional): number of input timesteps. Defaults to 1.
        T (int, optional): number of output timesteps. Defaults to 1.
    """
    with h5py.File(infFile, "r") as data:
        for iteration in range(len(data.keys())):
            time_in = []
            time_out = []
            ux = np.zeros((gridx, gridy, T_in))
            uy = np.zeros((gridx, gridy, T_in))
            vx = np.zeros((gridx, gridy, T))
            vy = np.zeros((gridx, gridy, T))
            vx_pred = np.zeros((gridx, gridy, T))
            vy_pred = np.zeros((gridx, gridy, T))
            p_in = np.zeros((gridx, gridy, T_in))
            p_out = np.zeros((gridx, gridy, T))
            p_pred = np.zeros((gridx, gridy, T))
            b_in = np.zeros((gridx, gridy, T_in))
            b_out = np.zeros((gridx, gridy, T))
            b_pred = np.zeros((gridx, gridy, T))      
            for index_in in range(T_in):
                time_in.append(data[f'inference_{iteration}/scales/sim_timein_{index_in}'])
                ux[:,:,index_in] = data[f'inference_{iteration}/tasks/input/velocity_{index_in}'][0,:]
                uy[:,:,index_in] = data[f'inference_{iteration}/tasks/input/velocity_{index_in}'][1,:]
                b_in[:,:,index_in] = data[f'inference_{iteration}/tasks/input/buoyancy_{index_in}'][:]
                p_in[:,:,index_in] = data[f'inference_{iteration}/tasks/input/pressure_{index_in}'][:]
            for index_out in range(T):
                time_out.append(data[f'inference_{iteration}/scales/sim_timeout_{index_out}'])
                vx[:,:,index_out] = data[f'inference_{iteration}/tasks/output/velocity_{index_out}'][0,:]
                vy[:,:,index_out] = data[f'inference_{iteration}/tasks/output/velocity_{index_out}'][1,:]
                b_out[:,:,index_out] = data[f'inference_{iteration}/tasks/output/buoyancy_{index_out}'][:]
                p_out[:,:,index_out] = data[f'inference_{iteration}/tasks/output/pressure_{index_out}'][:]
                vx_pred[:,:,index_out] = data[f'inference_{iteration}/tasks/model_output/velocity_{index_out}'][0,:]
                vy_pred[:,:,index_out] = data[f'inference_{iteration}/tasks/model_output/velocity_{index_out}'][1,:]
                b_pred[:,:,index_out] = data[f'inference_{iteration}/tasks/model_output/buoyancy_{index_out}'][:]
                p_pred[:,:,index_out] = data[f'inference_{iteration}/tasks/model_output/pressure_{index_out}'][:]

            # print(ux.shape, vy.shape, b_pred.shape, p_pred.shape)
            inferErrorPlot(ux, vx, vx_pred, uy, vy, vy_pred, b_in, b_out, b_pred, p_in, p_out, p_pred,
                time_in, time_out, dim, fno_path, gridx, gridy, rayleigh, prandtl)

def inference_write(iteration:int,
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
        for index_in in range(len(time_in)):
            data[f'inference_{iteration}/scales/sim_timein_{index_in}'] = time_in[index_in]
            data[f'inference_{iteration}/tasks/input/velocity_{index_in}'] = np.stack([ux[::xStep,::yStep, index_in], uy[::xStep,::yStep, index_in]], axis=0)
            data[f'inference_{iteration}/tasks/input/buoyancy_{index_in}'] = b_in[::xStep, ::yStep, index_in]
            data[f'inference_{iteration}/tasks/input/pressure_{index_in}'] = p_in[::xStep, ::yStep, index_in]
        for index_out in range(len(time_out)):
            data[f'inference_{iteration}/scales/sim_timeout_{index_out}']= time_out[index_out]
            if calc_loss and vx is not None:
                data[f'inference_{iteration}/tasks/output/velocity_{index_out}']= np.stack([vx[::xStep,::yStep, index_out], vy[::xStep,::yStep, index_out]], axis=0)
                data[f'inference_{iteration}/tasks/output/buoyancy_{index_out}'] = b_out[::xStep, ::yStep, index_out]
                data[f'inference_{iteration}/tasks/output/pressure_{index_out}']= p_out[::xStep, ::yStep, index_out]
            data[f'inference_{iteration}/tasks/model_output/velocity_{index_out}']= np.stack([vx_pred[::xStep,::yStep, index_out], vy_pred[::xStep,::yStep, index_out]], axis=0)
            data[f'inference_{iteration}/tasks/model_output/buoyancy_{index_out}']= b_pred[::xStep, ::yStep, index_out]
            data[f'inference_{iteration}/tasks/model_output/pressure_{index_out}']= p_pred[::xStep, ::yStep, index_out]
                        
def inference_with_loss(test_loader,loss,model,
                        gridx:int, gridy:int,
                        dim:str, nTest:int=1,
                        batch_size:int=1,tStep:int=1,
                        T_in:int=1,T:int=1,
                        device:str='cpu',y_normalizer=None):
    """
    Perform inference with loss calulations

    Args:
        test_loader (torch.DataLoader): test dataloader 
        loss (func): loss function
        model (torch.nn.Module) : FNO model
        gridx (int): x grid size
        gridy (int): y grid size
        dim (str): FNO2D or FNO3D strategy
        batch_size (int): inference batch size
        nTest (int, optional): number of test samples. Defaults to 1.
        tStep (int, optional): timestep slicing. Defaults to 1.
        T_in (int, optional): number of input timesteps. Defaults to 1.
        T (int, optional): number of output timesteps. Defaults to 1.
        device (str, optional): Defaults to 'cpu'.
        y_normalizer (func, optional): test data normalizer. Defaults to None.

    Returns:
        inputs (list): Dedalus input stack [velx, velz, buoyancy, pressure]
        outputs (list): Dedalus output stack [velx, velz, buoyancy, pressure]
        predictions (list): FNO output stack [velx, velz, buoyancy, pressure]
    """
   
    if dim == 'FNO3D':
        fno3d_loss = 0
    else:
        fno2d_full_loss = 0
        fno2d_step_loss = 0
        
    pred = torch.zeros([batch_size, gridx, gridy, T])
    inputs = []
    outputs = []
    predictions = []
        
    for step, (xx, yy) in enumerate(tqdm(test_loader)):
        xx, yy = xx.to(device), yy.to(device)
        xx_org = xx
        if dim == 'FNO3D':
            out = model(xx_org).view(batch_size, gridx, gridy, T)
            out = y_normalizer.decode(out)
            fno3d_loss += loss(out.view(1, -1), yy.view(1, -1)).item()
            pred = out 
        else:
            step_loss = 0
            for t in range(0, T, tStep):
                y = yy[..., t:t + tStep]   
                im = model(xx)
                step_loss += loss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
                
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)
        
                xx = torch.cat((xx[..., tStep:], im), dim=-1)
            
            fno2d_step_loss += step_loss.item()
            fno2d_full_loss += loss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
    
        inputs.append(xx_org)
        outputs.append(yy)
        predictions.append(pred)
        
        if step == 0 or step %10 == 0:
            print(f"Batch:{step} , xx:{xx_org.shape}, yy:{yy.shape}, pred:{pred.shape}")
            if dim == 'FNO3D':
                print(f"FNO3D loss: {fno3d_loss/((step+1)*batch_size)}")
            else:
                print(f"FNO2D step loss: {fno2d_step_loss/((step+1)*batch_size)/ (T/tStep)}")
                print(f"FNO2D full loss: {fno2d_full_loss/((step+1)*batch_size)}")
    
    return inputs, outputs, predictions

def inference_without_loss(model, input_data,
                          gridx:int, gridy:int,
                          dim:str,batch_size:int=1,
                          tStep:int=1,T_in:int=1,T:int=1,
                          device:str='cpu'):
    """
    Perform inference with loss calulations

    Args:
        model (torch.nn.Module) : FNO model
        input_data (torch.tensor): FNO input tensor stack [velx, velz, buoyancy, pressure]
        gridx (int): x grid size
        gridy (int): y grid size
        dim (str): FNO2D or FNO3D strategy
        batch_size (int): inference batch size
        tStep (int, optional): timestep slicing. Defaults to 1.
        T_in (int, optional): number of input timesteps. Defaults to 1.
        T (int, optional): number of output timesteps. Defaults to 1.
        device (str, optional): Defaults to 'cpu'.

    Returns:
        xx_org (torch.tensor): Dedalus input stack [velx, velz, buoyancy, pressure]
        pred (torch.tensor): FNO output stack [velx, velz, buoyancy, pressure]
    """
   
    
    pred = torch.zeros([batch_size, gridx, gridy, T])
    xx = input_data.to(device)
    xx_org = xx
    if dim == 'FNO3D':
        out = model(xx_org).view(batch_size, gridx, gridy, T)
        # out = y_normalizer.decode(out)
        pred = out 
    else:
        for t in range(0, T, tStep):
            im = model(xx)
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
    
            xx = torch.cat((xx[..., tStep:], im), dim=-1)
    

    return xx_org, pred
    
def model_inference(checkpoint:str, test_data_path:str, dim:str, modes:int, width:int,
                    gridx:int, gridy:int, batch_size:int, nTest:int=1,
                    xStep:int=1,yStep:int=1,tStep:int=1, index:int=0,
                    T_in:int=1, T:int=1, calc_loss:bool=False, loss=None,
                    device:str='cpu',single_data_path:bool=False, **kwargs):
    """
    Function to perform FNO model inference

    Args:
        checkpoint (str): path to FNO model checkpoint
        test_data_path (str): path to hdf5 test data file.
        dim (str): FNO2D or FNO3D strategy
        modes (int): number of fourier modes in the model
        width (int): number of neurons in the FNO model
        loss (func): loss function. Defaults to None
        calc_loss (bool): perform inference with loss
        gridx (int): x grid size
        gridy (int): y grid size
        batch_size (int): inference batch size
        xStep (int, optional): x-grid slicing. Defaults to 1.
        yStep (int, optional): y-grid slicing. Defaults to 1.
        tStep (int, optional): timestep slicing. Defaults to 1.
        index (int, optional): start index for T_in
        T_in (int, optional): number of input timesteps. Defaults to 1.
        T (int, optional): number of output timesteps. Defaults to 1.
        device (str, optional): Defaults to 'cpu'.
        nTest (int, optional): number of testing samples. Defaults to 1.
        single_data_path (bool): hdf5 file contains test and/or train data
        
        Optional:
        train_data_path (str): path to hdf5 train data file
        train_samples (int): number of training samples 

    Returns:
        inputs (np.ndarray): dedalus inputs
        predictions (np.ndarray): FNO outputs
        outputs: Dedalus outputs
    """
    
    y_normalizer = None
    inference_func_start = default_timer()
    print(f'Entered model_inference() at {inference_func_start}')

    # Model
    if dim == 'FNO3D':
        model = FNO3d(modes, modes, modes, width, T_in, T).to(device)
    else:
        model = FNO2d(modes, modes, width, T_in, T).to(device)

    model_checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    if 'model_state_dict' in model_checkpoint.keys():
        model.load_state_dict(model_checkpoint['model_state_dict'])
    else:
        model.load_state_dict(model_checkpoint)
    
    if 'train_samples' in kwargs.keys():
        train_samples = kwargs['train_samples']
    if 'train_data_path' in kwargs.keys():
        train_data_path = kwargs['train_data_path']
        
    # Inference
    print('Starting model inference...')
    inference_time_start = default_timer()

    model.eval()
    with torch.no_grad():
        if calc_loss:
            if single_data_path:
                if dim == 'FNO3D':
                    test_loader, y_normalizer = data_loading(dim, test_data_path, batch_size, gridx, gridy, xStep, yStep, tStep, T_in, T, nTest, index, single_data_path, train_samples=train_samples)
                else:
                    test_loader = data_loading(dim, test_data_path, batch_size, gridx, gridy, xStep, yStep, tStep, T_in, T, nTest, index, single_data_path)
            else:
                if dim == 'FNO3D':
                    test_loader, y_normalizer = data_loading(dim, test_data_path, batch_size, gridx, gridy, xStep, yStep, tStep,T_in, T, nTest, index, single_data_path, train_data_path=train_data_path, train_samples=train_samples)
                else:
                    test_loader = data_loading(dim, test_data_path, batch_size, gridx, gridy, xStep, yStep, tStep,T_in, T, nTest, index, single_data_path)
    
            inputs, outputs, predictions = inference_with_loss(test_loader, loss, model, 
                                                               gridx, gridy, dim,
                                                               nTest, batch_size, 
                                                               tStep, T_in, T, 
                                                               device,y_normalizer)
        else:
            test_reader = h5py.File(test_data_path, mode="r")
            dataloader_time_start = default_timer()
            input_data = torch.tensor(test_reader['test'][0, ::xStep, ::yStep, index: index + (T_in*tStep): tStep], dtype=torch.float)
            dataloader_time_stop = default_timer()
            print(f'Total time taken for dataloading (s): {dataloader_time_stop - dataloader_time_start}')
            inputs, predictions = inference_without_loss(model, input_data, 
                                                         gridx, gridy, dim, 
                                                         batch_size,
                                                         tStep, T_in, T, 
                                                         device)
          
    predictions_cpu = torch.stack(predictions).cpu()
    inference_time_stop = default_timer()
    print(f'Total time taken for model inference for {T} steps of {nTest} samples with batchsize {batch_size} on {device} (s): {inference_time_stop - inference_time_start}')
    
    inputs_cpu = torch.stack(inputs).cpu()
    inference_func_stop = default_timer()
    print(f'Exiting model_inference()...')
    print(f'Total time in model_inference() function (s): {inference_func_stop - inference_func_start}')
    
    if calc_loss:
        outputs_cpu = torch.stack(outputs).cpu()
        return np.array(inputs_cpu), np.array(outputs_cpu), np.array(predictions_cpu)
    else:
        return np.array(inputs_cpu), np.array(predictions_cpu)

def main( folder:str, dim:str, model_checkpoint:str, 
          modes:int, width:int, batch_size:int, 
          rayleigh:float, prandtl:float,
          gridx:int, gridy:int, 
          start_index:int, stop_index:int,
          timestep:int, dedalus_time_index:int,dt:int,
          test_data_path:str, train_data_path:str, single_data_path:str,
          nTest:int=1, train_samples=None,
          xStep:int=1, yStep:int=1,tStep:int=1,
          T_in:int=1, T:int=1,run:int=1,
          calc_loss:bool=False, plotFile:bool=False, store_result:bool=True,
          *args,**kwargs):
    """

    Args:
        folder (str): root path for storing inference results
        dim (str): FNO2D or FNO3D strategy
        model_checkpoint (str): path to model checkpoint
        modes (int): number of fourier modes
        width (int): number of neurons in layers
        batch_size (int): inference batch size 
        rayleigh (float):Rayleigh number 
        prandtl (float): Prandtl number 
        gridx (int): x grid size
        gridy (int): y grid size
        start_index (int): start index for input time
        stop_index (int): stop index for input time
        timestep (int): interval for input time
        dedalus_time_index (int): absolutime time index 
        dt (int): dedalus timestep
        calc_loss (bool): to calculate inference loss
        plotFile (bool): to plot crossectional plots from a Hdf5 file
        store_result (bool): to store inference result to a Hdf5 file 
        nTest (int, optional): number of inference samples. Defaults to 1.
        train_samples (int, optional): number of training samples. Defaults to None.
        xStep (int, optional): x-grid slicing. Defaults to 1.
        yStep (int, optional): y-grid slicing. Defaults to 1.
        tStep (int, optional): timestep slicing. Defaults to 1.
        T_in (int, optional): number of input timesteps. Defaults to 1.
        T (int, optional): number of output timesteps. Defaults to 1.
        run (int, optional): run tracker. Defaults to 1.
        single_data_path (str): path to hdf5 file containing test and/or train data
        test_data_path (str): path to hdf5 test data file.
        train_data_path (str): path to hdf5 train data file
    """
    
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    # since velx,vely,pressure and buoyacy are stacked
    gridx_state = 4*gridx
 
    fno_path = Path(f'{folder}/rbc_{dim}_N{nTest}_m{modes}_w{width}_bs{batch_size}_dt{dt}_tin{T_in}_inference_{device}_run{run}')
    fno_path.mkdir(parents=True, exist_ok=True)

    if plotFile:
        crosssection_plots(plotFile, gridx, gridy, dim, fno_path, rayleigh, prandtl, T_in, T)  
    else:
        for iteration, index in enumerate(range(start_index, stop_index, timestep)):
            start_index_org =  dedalus_time_index + index
            time_in, time_out = time_extract(start_index_org, dt, T_in, T, tStep)
            
            if calc_loss:
                if single_data_path:
                    inputs, outputs, predictions = model_inference(model_checkpoint, test_data_path, dim, modes, width,
                                                                    gridx_state, gridy, batch_size, nTest,
                                                                    xStep, yStep, tStep, index,
                                                                    T_in, T, calc_loss, LpLoss(size_average=False),
                                                                    device, single_data_path, train_samples=train_samples)
                else:
                    if dim =='FNO3D':
                        inputs, outputs, predictions = model_inference(model_checkpoint, test_data_path, dim, modes, width,
                                                                        gridx_state, gridy, batch_size, nTest,
                                                                        xStep, yStep, tStep, index,
                                                                        T_in, T, calc_loss, LpLoss(size_average=False),
                                                                        device, single_data_path, train_data_path=train_data_path, train_samples=train_samples)
                    else: 
                        inputs, outputs, predictions = model_inference(model_checkpoint, test_data_path, dim, modes, width,
                                                                        gridx_state, gridy, batch_size, nTest,
                                                                        xStep, yStep, tStep, index,
                                                                        T_in, T, calc_loss, LpLoss(size_average=False),
                                                                        device, single_data_path)
                    
                print(f"Model Inference: Input{inputs.shape}, Output{outputs.shape}, Prediction{predictions.shape}")
            else:
                inputs, predictions = model_inference(model_checkpoint, test_data_path, dim, modes, width,
                                                      gridx_state, gridy, batch_size, nTest,
                                                      xStep, yStep, tStep, index,
                                                      T_in, T, calc_loss, LpLoss(size_average=False),
                                                      device, single_data_path)
                print(f"Model Inference: Input{inputs.shape}, Prediction{predictions.shape}")
            
            # taking results for a random sample when batchsize > 1
            batches = predictions.shape[0]
            batchsize = predictions.shape[1]
            batch_num = np.random.randint(0,batches)
            sample = np.random.randint(0,batchsize)
            
            if dim == "FNO3D":
                # since test_a = test_a.reshape(nTest, gridx, gridy, 1, T_in).repeat([1,1,1,T,1])
                ux, uy, b_in, p_in = extract(inputs[batch_num, sample, :, :, 0, :], gridx, gridy, T_in)
            else:
                ux, uy, b_in, p_in = extract(inputs[batch_num, sample, :, :, :], gridx, gridy, T_in)
                
            vx_pred, vy_pred, b_pred, p_pred = extract(predictions[batch_num, sample, :, :, :], gridx, gridy,T)
           
            print(f"Inference for Batch Number: {batch_num}, Sample: {sample}") 
            if store_result:
                if calc_loss:
                    vx, vy, b_out, p_out = extract(outputs[batch_num, sample, :, :, :], gridx, gridy, T)
                    inference_write(iteration, ux, vx_pred, uy, vy_pred,
                                    b_in, b_pred, p_in, p_pred, 
                                    time_in, time_out,f'{fno_path}/inference.h5', calc_loss,
                                    xStep, yStep, tStep,
                                    vx, vy, b_out, p_out)
                    
                    inferErrorPlot(ux, vx, vx_pred, 
                                   uy, vy, vy_pred,
                                   b_in, b_out, b_pred,
                                   p_in, p_out, p_pred,
                                   time_in, time_out, dim, 
                                   fno_path, gridx, gridy, 
                                   rayleigh, prandtl)
                else:
                    inference_write(iteration, ux, vx_pred, uy, vy_pred,
                                    b_in, b_pred, p_in, p_pred, 
                                    time_in, time_out,f'{fno_path}/inference.h5', calc_loss,
                                    xStep, yStep, tStep)
            else:
                print(f'x-vel: {vx_pred} \n y-vel: {vy_pred} \n buoyancy: {b_pred} \n pressure: {p_pred}')
                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FNO Inference')
    parser.add_argument('--run', type=int, default=1,
                            help='training tracking number')
    parser.add_argument('--model_checkpoint', type=str,
                        help=" Torch model state path")
    parser.add_argument('--single_data_path', action='store_true',
                        help='if hdf5 file contain train, val and test data')
    parser.add_argument('--train_data_path', type=str,default=None,
                        help='path to train data hdf5 file')
    parser.add_argument('--test_data_path', type=str,default=None,
                        help='path to test data hdf5 file')
    parser.add_argument('--dim', type=str,default="FNO2D",
                        help="FNO2D+recurrent time or FNO3D")
    parser.add_argument('--modes', type=int, default=12,
                        help="Fourier modes")
    parser.add_argument('--width', type=int, default=20,
                        help="Width of layer")
    parser.add_argument('--batch_size', type=int, default=5,
                        help="Batch size")
    parser.add_argument('--rayleigh', type=float, default=1.5e7,
                        help="Rayleigh Number")
    parser.add_argument('--prandtl', type=float, default=1.0,
                        help="Prandtl Number")
    parser.add_argument('--gridx', type=int, default=256,
                        help="size of x-grid")
    parser.add_argument('--gridy', type=int, default=64,
                        help="size of y-grid")
    parser.add_argument('--T_in', type=int, default=1,
                        help='number of input timesteps to FNO')
    parser.add_argument('--T', type=int, default=1,
                        help='number of output timesteps to FNO')
    parser.add_argument('--dedalus_time_index', type=int, 
                        help='absolute time index for dedalus data')
    parser.add_argument('--start_index', type=int, 
                        help='relative time index for dedalus data')
    parser.add_argument('--stop_index', type=int, 
                        help='relative time index for dedalus data')
    parser.add_argument('--timestep', type=int, 
                        help='slicer for dedalus data')
    parser.add_argument('--dt', type=float, 
                        help='dedalus data dt')
    parser.add_argument('--train_samples', type=int, default=100,
                        help='Number of training samples')
    parser.add_argument('--nTest', type=int, default=1,
                            help='Number of test samples')
    parser.add_argument('--folder', type=str, default=os.getcwd(),
                            help='Path to which FNO model inference is saved')
    parser.add_argument('--plotFile', type=str, default=None,
                        help='path to inference data file')
    parser.add_argument('--store_result',action='store_true',
                        help='store inference result to hdf5 file')
    parser.add_argument('--calc_loss',action='store_true',
                        help='calculate inference loss')
    args = parser.parse_args()
    main(**args.__dict__)