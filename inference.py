""""
Perform inference and plot results for 2D Rayleigh Benard Convection

Usage:
    python inference.py \
        --run=<run_tracker> \
        --model=<checkpoint> \
        --train_data_path=<train_data_path> (only when using FNO3D) \
        --test_data_path=<test_data_path> (only when test and train data in multiple files) \
        --dim=FNO2D or FNO3D \
        --modes=12 \
        --width=20 \
        --batch_size=50 \
        --rayleigh=1.5e7 \
        --prandtl=1.0\
        --gridx=256 \
        --gridy=64 \
        --train_samples=<train_samples> (only when using FNO3D) \
        --test_samples=<test_samples> \
        --input_timesteps=<T_in> \
        --output_timesteps=<T> \
        --start_index=<dedalus_start_index> \
        --stop_index=<dedalus_stop_index> \
        --time_slice=<dedalus_time_slice> \
        --dedalus_time_index=<absolute_dedalus_time_index>
        --dt=<dedalus_data_dt>
        --folder=<results>  \
        
    optional args:
        --single_data_path=<path to hdf5 file containing train, val and test data>
        --plotFile (only to plot without model_inference)
        
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


parser = argparse.ArgumentParser(description='FNO Inference')
parser.add_argument('--run', type=int, default=1,
                        help='training tracking number')
parser.add_argument('--model', type=str,
                    help=" Torch model state path")
parser.add_argument('--single_data_path', type=str,default=None,
                        help='path to hdf5 file containing train, val and test data')
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
parser.add_argument('--input_timesteps', type=int, default=1,
                    help='number of input timesteps to FNO')
parser.add_argument('--output_timesteps', type=int, default=1,
                    help='number of output timesteps to FNO')
parser.add_argument('--dedalus_time_index', type=int, 
                    help='absolute time index for dedalus data')
parser.add_argument('--start_index', type=int, 
                    help='relative time index for dedalus data')
parser.add_argument('--stop_index', type=int, 
                    help='relative time index for dedalus data')
parser.add_argument('--time_slice', type=int, 
                    help='slicer for dedalus data')
parser.add_argument('--dt', type=float, 
                    help='dedalus data dt')
parser.add_argument('--train_samples', type=int, default=100,
                    help='Number of training samples')
parser.add_argument('--test_samples', type=int, default=1,
                        help='Number of test samples')
parser.add_argument('--folder', type=str, default=os.getcwd(),
                        help='Path to which FNO model inference is saved')
parser.add_argument('--plotFile', type=str, default=None,
                    help='path to inference data file')
args = parser.parse_args()

def extract(result, gridx, gridy, t):
    """
    unpacking stack [velx, velz, buoyancy, pressure]
    
    """
    ux = result[:gridx, :gridy, :t]
    uy = result[gridx:2*gridx, :gridy, :t]
    b = result[2*gridx:3*gridx, :gridy,:t]
    p = result[3*gridx:, :gridy,:t]
    
    # print(ux.shape, uy.shape, b.shape, p.shape)
    return ux, uy, b, p

def time_extract(time_index, t_in, t_out, dt, tStep):
    """
    Extracting simulation time from dedalus data
    """
    time_in = []
    for i in range(time_index, time_index + (t_in*tStep), tStep):
        time_in.append(i*dt)
    print("Input Time", time_in)
    time_out = []
    for j in range(time_index+t_in, time_index + (t_in+t_out)*tStep, tStep):
        time_out.append(j*dt)
    print("Output Time", time_out)
    return time_in, time_out

def inferErrorPlot( ux, vx, vx_pred,
                    uy, vy, vy_pred,
                    b_in, b_out, b_pred,
                    p_in, p_out, p_pred, 
                    time_out, time_in, 
                    dim, fno_path,
                    gridx, gridy,
                    rayleigh, prandtl):
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

def model_inference(args, *argv):
    inference_func_start = default_timer()
    print(f'Entered model_inference() at {inference_func_start}')

    if args.single_data_path is not None:
        test_data_path = train_data_path  = args.single_data_path
        test_reader = train_reader = h5py.File(train_data_path, mode="r")
    else:
        if args.dim =='FNO3D':
            train_data_path = args.train_data_path
            train_reader = h5py.File(train_data_path, mode="r")
        test_data_path = args.test_data_path
        test_reader = h5py.File(test_data_path, mode="r")
    # TODO : why do you need test data when evaluating the model ?

    dataloader_time_start = default_timer()
    print('Starting data loading....')
    if args.dim == 'FNO3D':
        train_a = torch.tensor(train_reader['train'][:train_samples, ::xStep, ::yStep, index: index + (T_in*tStep): tStep], dtype=torch.float)
        train_u = torch.tensor(train_reader['train'][:train_samples, ::xStep, ::yStep, index + (T_in*tStep):  index + (T_in + T)*tStep: tStep], dtype=torch.float)
    
    print(f"index: {index}")
    test_a = torch.tensor(test_reader['test'][:ntest, ::xStep, ::yStep, index: index + (T_in*tStep): tStep], dtype=torch.float)
    test_u = torch.tensor(test_reader['test'][:ntest, ::xStep, ::yStep, index + (T_in*tStep):  index + (T_in + T)*tStep: tStep], dtype=torch.float)
    dataloader_time_stop = default_timer()
    print(f'Total time taken for dataloading (s): {dataloader_time_stop - dataloader_time_start}')
    
    # Model
    if args.dim == 'FNO3D':
        a_normalizer = UnitGaussianNormalizer(train_a)
        test_a = a_normalizer.encode(test_a)
        y_normalizer = UnitGaussianNormalizer(train_u)
        test_u = y_normalizer.encode(test_u)
        test_a = test_a.reshape(ntest, gridx_state, gridy, 1, T_in).repeat([1,1,1,T,1])
        model = FNO3d(modes, modes, modes, width, T_in, T).to(device)
    else:
        model = FNO2d(modes, modes, width, T_in, T).to(device)

    checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
    if 'model_state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)
    print(f"Test input data:{test_a.shape}, Test output data: {test_u.shape}")

    # Inference
    print('Starting model inference...')
    inference_time_start = default_timer()

    pred = torch.zeros([batch_size, gridx_state, gridy, T])
    inputs = []
    outputs = []
    predictions = []
    fno2d_full_loss = 0
    fno2d_step_loss = 0
    fno3d_loss = 0
    
    model.eval()
    with torch.no_grad():
        for step, (xx, yy) in enumerate(tqdm(test_loader)):
            xx, yy = xx.to(device), yy.to(device)
            xx_org = xx
            if args.dim == 'FNO3D':
                out = model(xx_org).view(batch_size, gridx_state, gridy, T)
                out = y_normalizer.decode(out)
                fno3d_loss += myloss(out.view(1, -1), yy.view(1, -1)).item()
                pred = out 
            else:
                step_loss = 0
                for t in range(0, T, tStep):
                    y = yy[..., t:t + tStep]    # TODO : y is assigned but never used
                    im = model(xx)
                    step_loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
                    
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
             
                    xx = torch.cat((xx[..., tStep:], im), dim=-1)
                
                fno2d_step_loss += step_loss.item()
                fno2d_full_loss += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

            inputs.append(xx_org)
            outputs.append(yy)
            predictions.append(pred)
            if step == 0:
                print(f"Batch:{step} , xx:{xx_org.shape}, yy:{yy.shape}, pred:{pred.shape}")
                if args.dim == 'FNO3D':
                    print(f"FNO3D loss: {fno3d_loss/ntest}")
                else:
                    print(f"FNO2D step loss: {fno2d_step_loss/ ntest/ (T/tStep)}")
                    print(f"FNO2D full loss: {fno2d_full_loss/ ntest}")
         

    predictions_cpu = torch.stack(predictions).cpu()
    inference_time_stop = default_timer()
    print(f'Total time taken for model inference for {T} steps of {ntest} samples with batchsize {batch_size} on {device} (s): {inference_time_stop - inference_time_start}')
    
    inputs_cpu = torch.stack(inputs).cpu()
    outputs_cpu = torch.stack(outputs).cpu()

    inference_func_stop = default_timer()
    print(f'Exiting model_inference()...')
    print(f'Total time in model_inference() function (s): {inference_func_stop - inference_func_start}')
    
    return np.array(inputs_cpu), np.array(outputs_cpu), np.array(predictions_cpu)

# Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")

modes = args.modes
width = args.width
batch_size = args.batch_size
rayleigh = args.rayleigh 
prandtl = args.prandtl 
gridx = args.gridx
gridx_state = 4*gridx
gridy = args.gridy

T_in = args.input_timesteps
T = args.output_timesteps
start_index = args.start_index
stop_index = args.stop_index
timestep = args.time_slice
dedalus_time_index = args.dedalus_time_index
dt = args.dt

train_samples = args.train_samples
ntest = args.test_samples

xStep = 1
yStep = 1
tStep = 1

run = args.run
fno_path = Path(f'{args.folder}/rbc_{args.dim}_N{ntest}_m{modes}_w{width}_bs{batch_size}_dt{dt}_tin{T_in}_inference_{device}_run{run}')
fno_path.mkdir(parents=True, exist_ok=True)

    fno_path = Path(f'{args.folder}/rbc_{args.dim}_N{ntest}_m{modes}_w{width}_bs{batch_size}_inference_{device}')
    fno_path.mkdir(parents=True, exist_ok=True)

if args.plotFile is not None:
    infFile = args.plotFile
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
                time_out, time_in, args.dim, fno_path, gridx, gridy, rayleigh, prandtl)
else:
    for iteration, index in enumerate(range(start_index, stop_index, timestep)):
        start_index_org =  dedalus_time_index + index
        time_in, time_out = time_extract(start_index_org, T_in, T, dt, tStep)
        
        inputs, outputs, predictions = model_inference(args, index)
        print(f"Model Inference: Input{inputs.shape}, Output{outputs.shape}, Prediction{predictions.shape}")
        
        # taking results for a random sample when batchsize > 1
        batches = predictions.shape[0]
        batchsize = predictions.shape[1]
        batch_num = np.random.randint(0,batches)
        sample = np.random.randint(0,batchsize)
        
        if args.dim == "FNO3D":
            # since test_a = test_a.reshape(ntest, gridx, gridy, 1, T_in).repeat([1,1,1,T,1])
            ux, uy, b_in, p_in = extract(inputs[batch_num, sample, :, :, 0, :], gridx, gridy, T_in)
        else:
            ux, uy, b_in, p_in = extract(inputs[batch_num, sample, :, :, :], gridx, gridy, T_in)
            
        vx_pred, vy_pred, b_pred, p_pred = extract(predictions[batch_num, sample, :, :, :], gridx, gridy,T)
        vx, vy, b_out, p_out = extract(outputs[batch_num, sample, :, :, :], gridx, gridy, T)

        # Storing inference result
        with h5py.File(f'{fno_path}/inference.h5', "a") as data:
            for index_in in range(len(time_in)):
                data[f'inference_{iteration}/scales/sim_timein_{index_in}'] = time_in[index_in]
                data[f'inference_{iteration}/tasks/input/velocity_{index_in}'] = np.stack([ux[::xStep,::yStep, index_in], uy[::xStep,::yStep, index_in]], axis=0)
                data[f'inference_{iteration}/tasks/input/buoyancy_{index_in}'] = b_in[::xStep, ::yStep, index_in]
                data[f'inference_{iteration}/tasks/input/pressure_{index_in}'] = p_in[::xStep, ::yStep, index_in]
            for index_out in range(len(time_out)):
                data[f'inference_{iteration}/scales/sim_timeout_{index_out}']= time_out[index_out]
                data[f'inference_{iteration}/tasks/output/velocity_{index_out}']= np.stack([vx[::xStep,::yStep, index_out], vy[::xStep,::yStep, index_out]], axis=0)
                data[f'inference_{iteration}/tasks/output/buoyancy_{index_out}'] = b_out[::xStep, ::yStep, index_out]
                data[f'inference_{iteration}/tasks/output/pressure_{index_out}']= p_out[::xStep, ::yStep, index_out]
                data[f'inference_{iteration}/tasks/model_output/velocity_{index_out}']= np.stack([vx_pred[::xStep,::yStep, index_out], vy_pred[::xStep,::yStep, index_out]], axis=0)
                data[f'inference_{iteration}/tasks/model_output/buoyancy_{index_out}']= b_pred[::xStep, ::yStep, index_out]
                data[f'inference_{iteration}/tasks/model_output/pressure_{index_out}']= p_pred[::xStep, ::yStep, index_out]
        
        print(f"Plotting Batch Number: {batch_num}, Sample: {sample}")  
        inferErrorPlot(ux, vx, vx_pred, uy, vy, vy_pred, b_in, b_out, b_pred, p_in, p_out, p_pred,
                time_out, time_in, args.dim, fno_path, gridx, gridy, rayleigh, prandtl)
        