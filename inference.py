""""
Perform inference and plot results for 2D Rayleigh Benard Convection

Usage:
    python inference.py \
             --model=<checkpoint> \
             --data_path=<test_data> \
             --dim=FNO2D or FNO3D \
             --modes=12 \
             --width=20 \
             --batch_size=50 \
             --folder=results  \
             --time_file=<dedalus_datafile> \
             --plotFile

"""

import os
import sys
import functools
import operator
import h5py
import math
import copy
import scipy
import pickle
import scipy.io
import argparse
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from timeit import default_timer
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary    # TODO: is it really needed ?
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from utils import CudaMemoryDebugger, format_tensor_size, LpLoss, UnitGaussianNormalizer, rbc_data
from fno3d import FNO3d
from fno2d_recurrent import FNO2d

parser = argparse.ArgumentParser(description='Inference')
parser.add_argument('--model', type=str,
                    help=" Torch model state path")
parser.add_argument('--data_path', type=str,
                    help='path to data')
parser.add_argument('--dim', type=str,default="FNO2D",
                    help="FNO2D+recurrent time or FNO3D")
parser.add_argument('--modes', type=int, default=12,
                    help="Fourier modes")
parser.add_argument('--width', type=int, default=20,
                    help="Width of layer")
parser.add_argument('--batch_size', type=int, default=5,
                    help="Batch size")
parser.add_argument('--folder', type=str, default=os.getcwd(),
                        help='Path to which FNO model inference is saved')
parser.add_argument('--time_file', type=str,
                    help="Dedalus simulation file to extract time")
parser.add_argument('--plotFile', action="store_true",
                    help='Plot from inference data file')
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")
if device == 'cuda':
    torch.cuda.empty_cache()
    memory = CudaMemoryDebugger(print_mem=True)

def extract(result, gridx, gridz, t):
    ux = result[:gridx, :gridz, :t]
    uz = result[gridx:2*gridx, :gridz, :t]
    b = result[2*gridx:3*gridx, :gridz,:t]
    p = result[3*gridx:, :gridz,:t]
    print(ux.shape, uz.shape, b.shape, p.shape)
    return ux, uz, b, p

def time_extract(file, start_index, t_in, t_out):
    time_in = []
    for i in range(start_index, start_index+t_in):
        # print(rbc_data(file,i, False, True))
        _,_,time,_,_ = rbc_data(file,i, False, True)
        time_in.append(time)
    print("Input", time_in)
    time_out = []
    for i in range(start_index+t_in, start_index+t_in+t_out):
        # print(rbc_data(file,i, False, True))
        _,_,time,_,_ = rbc_data(file,i, False, True)
        time_out.append(time)
    print("Output", time_out)
    return time_in, time_out

def inferPlot(ux, vx, vx1,
              uz, vz, vz1,
              b_in, b_out, b_out1,
              p_in, p_out, p_out1, 
              time_out, time_in, 
              dim, fno_path,
              gridx, gridz):
    for t in range(len(time_out)):
        row = 2
        col = 4
        xStep = 10
        zStep = 10
        fig = plt.figure(figsize=(16, 12))
        ax1 = fig.add_subplot(row, col, 1)
        # ax1.set_title(r'Input $u(x,z)_{t}$ Vs Dedalus $(U(x,z)_{t+10})$ Vs. FNO2D_time$(U(x,z)_{t+10})$')
        ax1.set_title(fr'Input $u(x)$ at t={np.round(time_in[t],2)}')
        ax1.plot(ux[::xStep,::zStep,t],color='b',marker ='o',label="ux")
        ax1.grid()


        ax2 = fig.add_subplot(row,col, 2)
        ax2.set_title(fr'Output $u(x)$ at t={np.round(time_out[t],2)}')
        ax2.plot(vx[::xStep,::zStep,t],color='g',marker ='o',label="ded-vx")
        ax2.plot(vx1[::xStep,::zStep,t],color='r',marker ='o',ls='--',label="fno-vx")
        ax2.grid()

        # ax9.errorbar(x, np.average(vx[::xStep,::zStep,t],axis=1), yerr=np.average(vx_error, axis=1), marker='o',color='purple',capsize=3, markersize=6,linestyle='none')
        # ax9.set_ylabel(r"$\overline{|vx_{ded}-vx_{fno}|}_{z}$")
        # ax9.set_xlabel("X Grid")

        ax3 = fig.add_subplot(row, col, 3)
        ax3.set_title(fr'Input $u(z)$ at t={np.round(time_in[t],2)}')
        ax3.plot(uz[::xStep,::zStep,t],color='b',marker ='o',label="uz")
        ax3.grid()

        ax4 = fig.add_subplot(row,col, 4)
        ax4.set_title(fr'Output $u(z)$ at t={np.round(time_out[t],2)}')
        ax4.plot(vz[::xStep,::zStep,t],marker ='o',color='g',label="ded-vz")
        ax4.plot(vz1[::xStep,::zStep,t],marker ='o',color='r',linestyle='--',label="fno-vz")
        ax4.grid()

        ax5 = fig.add_subplot(row, col, 5)
        ax5.set_title(fr'Input Pressure $p(x,z)$ at t={np.round(time_in[t],2)}')
        ax5.plot(p_in[::xStep,::zStep,t],color='b',marker ='o',label="p_in")
        ax5.grid()

        ax6 = fig.add_subplot(row,col, 6)
        ax6.set_title(fr'Output Pressure $p(x,z)$ at t={np.round(time_out[t],2)}')
        ax6.plot(p_out[::xStep,::zStep,t],marker ='o',color='g',label="ded-p")
        ax6.plot(p_out1[::xStep,::zStep,t],marker ='o',color='r',linestyle='--',label="fno-p")
        ax6.grid()

        ax7 = fig.add_subplot(row,col, 7)
        ax7.set_title(fr'Input Buoyancy $b(x,z)$ at t={np.round(time_in[t],2)}')
        ax7.plot(b_in[::xStep,::zStep,t],marker ='o',color='b',label="b_in")
        ax7.grid()

        ax8 = fig.add_subplot(row,col, 8)
        ax8.set_title(fr'Output Buoyancy $b(x,z)$ at t={np.round(time_out[t],2)}')
        ax8.plot(b_out[::xStep,::zStep,t],marker ='o',color='g',label="ded-b")
        ax8.plot(b_out1[::xStep,::zStep,t],marker ='o',linestyle='--',color='r',label="fno-b")
        ax8.grid()

        fig.suptitle(f'RBC-2D with {gridx}'+r'$\times$'+f'{gridz} grid and $Ra=10^7, Pr=1$ using FN03D')#\n *(FNO=green["."],Dedalus=["-"])')
        ded_patch = Line2D([0], [0], label='Dedalus',marker='o', color='g')
        fno_patch = Line2D([0], [0], label='FNO',marker='o', linestyle='--', color='r')
        fig.legend(handles=[ded_patch, fno_patch], loc="upper right")
        fig.tight_layout()
        fig.show()
        fig.savefig(f"{fno_path}/{dim}_NX{gridx}_NZ{gridz}_{t}.pdf")

def inferErrorPlot( ux, vx, vx1,
                    uz, vz, vz1,
                    b_in, b_out, b_out1,
                    p_in, p_out, p_out1, 
                    time_out, time_in, 
                    dim, fno_path,
                    gridx, gridz):
    for t in range(len(time_out)):
        row = 2
        col = 4
        xStep = 30
        zStep = 30
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
        ax1.plot(x,ux[::xStep,::zStep,t],color='b',marker ='o',label="ux")
        ax1.plot(x,vx[::xStep,::zStep,t],color='g',marker ='o',label="ded-vx")
        ax1.plot(x,vx1[::xStep,::zStep,t],color='r',marker ='o',ls='--',label="fno-vx")
        # ax1.set_ylabel("Z grid")
        ax1.grid()

        ax2.set_title(fr'Velocity: $u(z)$ ')
        ax2.plot(x,uz[::xStep,::zStep,t],color='b',marker ='o',label="uz")
        ax2.plot(x,vz[::xStep,::zStep,t],marker ='o',color='g',label="ded-vz")
        ax2.plot(x,vz1[::xStep,::zStep,t],marker ='o',color='r',linestyle='--',label="fno-vz")
        # ax2.set_ylabel("Z grid")
        ax2.grid()

        ax3.set_title(fr'Pressure: $p(x,z)$')
        ax3.plot(x,p_in[::xStep,::zStep,t],color='b',marker ='o',label="p_in")
        ax3.plot(x,p_out[::xStep,::zStep,t],marker ='o',color='g',label="ded-p")
        ax3.plot(x,p_out1[::xStep,::zStep,t],marker ='o',color='r',linestyle='--',label="fno-p")
        # ax3.set_ylabel("Z grid")
        ax3.grid()

        ax4.set_title(fr'Buoyancy: $b(x,z)$')
        ax4.plot(x,b_in[::xStep,::zStep,t],marker ='o',color='b',label="b_in")
        ax4.plot(x,b_out[::xStep,::zStep,t],marker ='o',color='g',label="ded-b")
        ax4.plot(x,b_out1[::xStep,::zStep,t],marker ='o',linestyle='--',color='r',label="fno-b")
        # ax4.set_ylabel("Z grid")
        ax4.grid()

        ax5.errorbar(x, np.average(vx[::xStep,::zStep,t],axis=1), yerr=np.average(np.abs(vx1[::xStep,::zStep,t]-vx[::xStep,::zStep,t]), axis=1), marker='o',color='purple',capsize=3, markersize=6,linestyle='none')
        ax5.set_ylabel(r"$\overline{|vx_{ded}-vx_{fno}|}_{z}$")
        ax5.set_xlabel("X Grid")

        ax6.errorbar(x, np.average(vz[::xStep,::zStep,t],axis=1), yerr=np.average(np.abs(vz1[::xStep,::zStep,t]-vz[::xStep,::zStep,t]), axis=1), marker='o',color='purple',capsize=3, markersize=6,linestyle='none')
        ax6.set_ylabel(r"$\overline{|vz_{ded}-vz_{fno}|}_{z}$")
        ax6.set_xlabel("X Grid")

        ax7.errorbar(x, np.average(p_out[::xStep,::zStep,t],axis=1), yerr=np.average(np.abs(p_out1[::xStep,::zStep,t]-p_out[::xStep,::zStep,t]), axis=1), marker='o',color='purple',capsize=3, markersize=6,linestyle='none')
        ax7.set_ylabel(r"$\overline{|p_{ded}-p_{fno}|}_{z}$")
        ax7.set_xlabel("X Grid")

        ax8.errorbar(x, np.average(b_out[::xStep,::zStep,t],axis=1), yerr=np.average(np.abs(b_out1[::xStep,::zStep,t]-b_out[::xStep,::zStep,t]), axis=1), marker='o',color='purple',capsize=3, markersize=6,linestyle='none')
        ax8.set_ylabel(r"$\overline{|b_{ded}-b_{fno}|}_{z}$")
        ax8.set_xlabel("X Grid")

        fig.suptitle(f'RBC-2D with {gridx}'+r'$\times$'+f'{gridz} grid and $Ra=10^7, Pr=1$ using {dim}')  
        ded_patch = Line2D([0], [0], label=f'Dedalus at t={np.round(time_out[t],2)}',marker='o', color='g')
        fno_patch = Line2D([0], [0], label=f'FNO at t={np.round(time_out[t],2)}',marker='o', linestyle='--', color='r')
        inp_patch = Line2D([0], [0], label=f'Input at t={np.round(time_in[t],2)}',marker='o', color='b')
        fig.legend(handles=[inp_patch,ded_patch, fno_patch], loc="upper right")
        fig.tight_layout()
        fig.show()
        # fig.savefig(f"{fno_path}/{dim}_NX{gridx}_NZ{gridz}_error_{t}.pdf")
        fig.savefig(f"{fno_path}/{dim}_NX{gridx}_NZ{gridz}_error_{t}.png")

def model_inference(data_path):
    reader = h5py.File(data_path, mode="r")
    train_a = torch.tensor(reader['train'][:ntrain, ::xStep, ::zStep, start_index: start_index+T_in],dtype=torch.float)
    train_u = torch.tensor(reader['train'][:ntrain, ::xStep, ::zStep, start_index+T_in:T+start_index+T_in], dtype=torch.float)
    test_a = torch.tensor(reader['test'][:ntest, ::xStep, ::zStep, start_index: start_index+T_in], dtype=torch.float)
    test_u = torch.tensor(reader['test'][:ntest, ::xStep, ::zStep, start_index+T_in: T+start_index+T_in], dtype=torch.float)


    # Model
    if args.dim == 'FNO3D':
        a_normalizer = UnitGaussianNormalizer(train_a)
        test_a = a_normalizer.encode(test_a)
        y_normalizer = UnitGaussianNormalizer(train_u)
        test_u = y_normalizer.encode(test_u)
        test_a = test_a.reshape(ntest, gridx, gridz, 1, T_in).repeat([1,1,1,T,1])
        model = FNO3d(modes, modes, modes, width).to(device)
    else:
        model = FNO2d(modes, modes, width).to(device)

    model.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))   
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)
    print(f"Test input data:{test_a.shape}, Test output data: {test_u.shape}")

    # Inference
    pred = torch.zeros([iterations, batch_size, gridx, gridz, T])
    index = 0
    inputs = []
    outputs = []
    predictions = []
    with torch.no_grad():
        # with open(f'{fno_path}/info.txt', 'w') as file:
        for step, (xx, yy) in enumerate(test_loader):
            test_l2 = 0
            xx, yy = xx.to(device), yy.to(device)
            xx_org = xx
            if args.dim == 'FNO3D':
                out = model(xx).view(batch_size, gridx, gridz, T)
                # print(out.shape)
                out = y_normalizer.decode(out)
                test_l2 += myloss(out.view(1, -1), yy.view(1, -1)).item()
                pred[step]= out
            else:
                for t in range(0, T, tStep):
                    y = yy[..., t:t + tStep]
                    im = model(xx)
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                    # file.write(f"diff at {t}: {xx-im},\n im: {im.shape}\n")
                    # file.write(f"pred={pred.shape},\n{pred}\n")

                    xx = torch.cat((xx[..., tStep:], im), dim=-1)
                test_l2 += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

            inputs.append(xx_org)
            outputs.append(yy)
            predictions.append(pred)
            
            # file.write(f"index:{step} , xx:{xx.shape}, yy:{yy.shape}, pred:{np.array(predictions).shape},loss:{test_l2}, \
            #             \nprediction: {predictions},\n \
            #             \ninput: {xx_org}\n, \
            #             \ndiff:{np.array(predictions)-np.array(xx_org)}\n")

    # Storing inference result
    with h5py.File(f'{fno_path}/inference.h5', "w") as data:
        data['input'] = inputs
        data['output'] = outputs
        data['prediction'] = predictions
    data.close()
    return np.array(inputs), np.array(outputs), np.array(predictions)

# Config
modes = args.modes
width = args.width
batch_size = args.batch_size

gridx = 4*256
gridz = 64

xStep = 1
zStep = 1
tStep = 1

start_index = 500
T_in = 10
T = 10
time_in, time_out = time_extract(args.time_file, start_index, T_in, T)

myloss = LpLoss(size_average=False)

# Load data
ntrain = 100
ntest = 50
iterations = int(ntest/batch_size)

fno_path = Path(f'{args.folder}/rbc_{args.dim}_N{ntest}_m{modes}_w{width}_bs{batch_size}_inference')
fno_path.mkdir(parents=True, exist_ok=True)


if args.plotFile:
    infFile = f'{fno_path}/inference.h5'
    with h5py.File(infFile, "r") as data:
        inputs = data['input'][:]
        outputs = data['output'][:]
        predictions = data ['prediction'][:]
    data.close()
else:
    inputs, outputs, predictions = model_inference(args.data_path)


## Plotting
sample_size = predictions.shape[0]
batch_sample = predictions.shape[1]
sample = np.random.randint(0,sample_size)
batch = np.random.randint(0,batch_sample)

if args.dim == "FNO3D":
    ux, uz, b_in, p_in = extract(inputs[sample, batch,:,:,0,:], gridx//4, gridz, T_in)
    vx1, vz1, b_out1, p_out1 = extract(predictions[0,sample,batch,:,:], gridx//4, gridz, T)
else:
    ux, uz, b_in, p_in = extract(inputs[sample, batch,:,:], gridx//4, gridz, T_in)
    vx1, vz1, b_out1, p_out1 = extract(predictions[sample,batch,:,:], gridx//4, gridz,T)
vx, vz, b_out, p_out = extract(outputs[sample,batch,:,:], gridx//4, gridz, T)

inferErrorPlot(ux, vx, vx1,uz, vz, vz1,b_in, b_out, b_out1,p_in, p_out, p_out1,
               time_out, time_in, args.dim, fno_path ,gridx//4, gridz)