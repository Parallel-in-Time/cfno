#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
sys.path.insert(2, os.getcwd())
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from timeit import default_timer
import torch
from cfno.utils import readConfig
from cfno.data.preprocessing import HDF5Dataset
from cfno.training.pySDC import FourierNeuralOp
from cfno.simulation.post import computeMeanSpectrum, getModes, contourPlotStraat
from cfno.simulation.post import OutputFiles
from cfno.utils import compile_timing
from cfno.models.cfno2d import CFNO2D

torch.set_float32_matmul_precision("high")  # Enable TF32 matmul
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch._dynamo.config.cache_size_limit = 128 


# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Evaluate a model on a given dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--dataFile", default="dataset.h5", help="name of the dataset HDF5 file after preprocessing")
parser.add_argument(
    "--tSteps", default="1",type=int, help="number of autoregressive steps")
parser.add_argument(
    "--batchsize", default="1",type=int, help="number of samples")
parser.add_argument(
    "--model_dt", default="1e-3", type=float, help="model timestep")
parser.add_argument(
    "--checkpoint", default="model.pt", help="name of the file storing the model")
parser.add_argument(
    "--iSimu", default=15, type=int, help="index of the simulation to eval with")
parser.add_argument(
    "--imgExt", default="png", help="extension for figure files")
parser.add_argument(
    "--ndim", default=2, type=int, help="FNO2D or 3D")
parser.add_argument(
    "--data_aug",  action="store_true", help='Add noisy data per batch while training')
parser.add_argument(
    "--model_class", default="CFNO", help="CFNO or FNO")
parser.add_argument(
    "--sampling", default="random", help="random or sequential sampling")
parser.add_argument("--compile", action="store_true", 
                    help="using torch.compile for inference in default mode")
parser.add_argument(
    "--loss_axis", default=2, type=int, help="loss axis")
parser.add_argument(
    "--evalDir", default="eval", help="directory to store the evaluation results")
parser.add_argument(
    "--config", default=None, help="configuration file")
args = parser.parse_args()

# -------------------------------------------------------------------------
# -- Relative error over time
# -------------------------------------------------------------------------
def norm(x, ndim=2, loss_axis=3):
    if ndim == 2:
        axis = (-2,-1)
        value = np.linalg.norm(x, axis=axis)
    else:
        if loss_axis == 2:
            axis = (-2,-1)
            value = np.linalg.norm(x, axis=axis)
        else:
            axis = (-3,-2,-1)
            # value = np.sqrt(np.sum(x**2, axis=axis))
            x_tensor = torch.from_numpy(x)
            value = torch.linalg.vector_norm(x_tensor, dim=axis).cpu().detach().numpy()
       
    return value

def computeError(uPred, uRef, ndim, loss_axis):
    diff = norm(uRef-uPred, ndim, loss_axis)
    nRef = norm(uRef, ndim, loss_axis)
    return diff/nRef

def sliceToStr(s:slice):
    out = ":"
    if s.start is not None:
        out = str(s.start)+out
    if s.stop is not None:
        out = out+str(s.stop)
    return out

# Create summary file, and write header
def fmt(hdfFloat): return float(hdfFloat[()])

def contourPlot2(field, x, y, title=None,  
                 saveFig=False, closeFig=True, time=None):
    
    fig, axs = plt.subplots(1, constrained_layout=False)
    # scales =  (np.min(field), np.max(field))

    def setup(ax):
        ax.axis('on')
        ax.set_aspect(1.5, adjustable='box')
        xticks = [0, 3.14, 2*3.14]
        xlabels = ['0', r'$\pi$', r'2$\pi$']
        ax.set_xticks(xticks, labels=xlabels)
        yticks = [-1, 0, 1]
        ylabels = [1, 0, -1]
        ax.set_yticks(yticks, labels=ylabels)


    def setColorbar(im, ax):
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size = "2%", pad=1e-10)
        pos = ax.get_position()
        cax = fig.add_axes([
                pos.x1+0.08 ,  
                pos.y0-0.04,          
                0.02,           
                pos.y1-pos.y0+0.1  
            ])
        im.cmap.set_under("white")
        im.cmap.set_over("white")
        # im.set_clim(1.0, 2.0)
        cbar = fig.colorbar(im, cax=cax) #ticks = yticks
        cbar.ax.set_yticks([1.0, 1.5, 2.0])
        cbar.ax.set_yticklabels(['1', '1.5', '2'])


    timeSuffix = f' at t = {np.round(time,3)}s' if time is not None else ''
  
    im0 = axs.pcolormesh(x, y, field, cmap='RdBu_r', rasterized=True)
    fig.canvas.draw()
    setup(axs)
    setColorbar(im0, axs)
    axs.set_title(f'{title}{timeSuffix}', fontsize=14)
    fig.tight_layout()
    if saveFig:
        plt.savefig(f'{saveFig}', bbox_inches='tight', pad_inches=0, dpi=200)
    if closeFig:
        plt.close(fig)

if args.config is not None:
    config = readConfig(args.config)
    if "eval" in config:
        args.__dict__.update(**config["eval"])
    if "data" in config and "dataFile" in config["data"]:
        args.dataFile = config.data.dataFile
    if "train" in config and "checkpoint" in config["train"]:
        args.checkpoint = config.train.checkpoint
        args.ndim = config.train.ndim
        args.__dict__.update(**config.train)
        if "trainDir" in config.train:
            FourierNeuralOp.TRAIN_DIR = config.train.trainDir

dataFile = args.dataFile
checkpoint = args.checkpoint
iSimu = args.iSimu
imgExt = args.imgExt
evalDir = args.evalDir
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
tSteps = args.tSteps
model_dt = args.model_dt
sampling = args.sampling
loss_axis = args.loss_axis
inSize = args.batchsize
ndim = args.ndim

if ndim == 3:
    # u0_50 = np.empty((1,4,50,96,64))
    # uRef_50 = np.empty((1,4,50,96,64))
    # uRef_single50 = np.empty((tSteps,1,4,50,96,64))
    # u0_50 = np.empty((50,4,1,96,64))
    # uRef_50 = np.empty((50,4,1,96,64))
    # uRef_single50 = np.empty((tSteps,50,4,1,96,64))
    u0_50 = np.empty((5,4,inSize,96,64))
    uRef_50 = np.empty((5,4,inSize,96,64))
    uRef_single50 = np.empty((tSteps,5,4,inSize,96,64))
else:
    u0_50 = np.empty((50,4,96,64))
    uRef_50 = np.empty((50,4,96,64))
    uRef_single50 = np.empty((tSteps,50,4,96,64))
time_list50 = []
# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
dataset = HDF5Dataset('dedalus_data/dataset_96x64_Ra5e6_dt0_5_sol.h5')
dedalus_dataFile = OutputFiles('Straat_data/simu_720966/run_data')
time_org = dedalus_dataFile.file(0)['scales']['sim_time']
os.makedirs(evalDir, exist_ok=True)
nSamples = dataset.infos["nSamples"][()]
batchsize = args.batchsize 
nSimu = dataset.infos["nSimu"][()]
dtInput = fmt(dataset.infos["dtInput"])
dtSample = fmt(dataset.infos["dtSample"])
dtData = fmt(dataset.infos["dtData"])
error_list = []
single_step_error_list = []


if args.compile:
    chk = torch.load(checkpoint, map_location=device)
    model = CFNO2D(**chk['model'])
    print("Using torch.compile(model, mode=default) for inference")
    model.compile(mode='default')
    model.eval()
else:
    model = FourierNeuralOp(checkpoint=checkpoint, model_class=args.model_class, ndim=ndim, data_aug=False)


HEADER = """
# FNO Data for Straat Paper 

- model name: {checkpoint}
- dataset : {dataFile}
    - nSamples : {nSamples}
    - dtInput (between input and output of the model) : {dtInput}
    - dtSample (between two samples) : {dtSample}
    - outType : {outType}
    - outScaling : {outScaling}
    --batchSize: {batchsize}
    --tSteps: {tSteps}

"""
op = os.path
with open(op.dirname(op.abspath(op.realpath(__file__)))+"/eval_template_straat.md") as f:
    TEMPLATE = f.read()
summary = open(f"{evalDir}/eval_straat.md", "a")
summary.write(HEADER.format(
    checkpoint=checkpoint, dataFile=dataFile, nSamples=nSamples,
    dtInput=dtInput, dtSample=dtSample,
    outType=dataset.outType, outScaling=dataset.outScaling, batchsize=batchsize, tSteps=tSteps))

# Performing inference on different validation sets
val_set = [15,16,17,18,19,20]
# val_set = [16]
for index,iVal in enumerate(val_set):
    if iVal < 20:
        assert iVal < nSimu, f"cannot evaluate with iSimu={iVal} with only {nSimu} simu"
        indices = slice(iVal*nSamples, (iVal+1)*nSamples)
        print(f'Running inference using {iVal} dataset....')
        # Initial solution for all samples
        u0_full = dataset.inputs[indices]
        # Reference solution for all samples
        uRef_full = dataset.outputs[indices].copy()
        if dataset.outType == "update":
            uRef_full /= dataset.outScaling
            uRef_full += u0_full
            
        if ndim == 3:
            # u0_full = u0_full[:,:,:,:,np.newaxis,]
            # uRef_full = uRef_full[:,:,:,:,np.newaxis]
            u0_full = u0_full[:,:,np.newaxis,:,:]
            uRef_full = uRef_full[:,:,np.newaxis,:,:]
    
        if args.sampling == 'random':
            # input solution of batchsize
            start_idx = np.sort(np.random.choice(np.arange(1000,2000,5), size=batchsize, replace=False))
            # np.array([1025, 1070, 1150, 1380, 1405, 1410, 1605, 1720, 1725, 1730])
            # np.sort(np.random.choice(np.arange(1000,2000,5), size=batchsize, replace=False))
            print(f'input index: {start_idx}')
            print(f'input time_range: {time_org[start_idx]}')
            u0 = u0_full[start_idx]
            # output solution of batchsize
            out_index = start_idx + int((tSteps*model_dt)/dtData)
            print(f'Ref index: ({out_index}, \
                    output time_range: {time_org[out_index]}')
            uRef = u0_full[out_index]
            out_index_start = out_index[0]
        else:
            start_idx = 1000
            end_idx = int(start_idx + batchsize)
            print(f'input index: {start_idx, end_idx}')
            print(f' input time_range: {time_org[start_idx:end_idx:1]}')
            u0 = u0_full[start_idx: end_idx: 1]
            out_index_start = start_idx + int((tSteps*model_dt)/dtData)
            out_index_stop = end_idx + int((tSteps*model_dt)/dtData)
            print(f'Ref index: ({out_index_start},{out_index_stop}), \
            output time_range: {time_org[out_index_start: out_index_stop: 1]}')
            uRef = u0_full[out_index_start: out_index_stop: 1]
        time_list50.append(out_index_start)
        # full domain evaluation
        if ndim == 2:
            slices = (u0.shape[-2], u0.shape[-1])
            u0_50[index*inSize: (index+1)*inSize,:,:,:] = u0
            uRef_50[index*inSize: (index+1)*inSize,:,:,:] = uRef
        else:
            u0 = np.moveaxis(u0, [0,2], [2,0])
            uRef = np.moveaxis(uRef, [0,2], [2,0])
            print(f'u0_full 3d shape: {u0.shape}')
            print(f'uRef_full 3d shape: {uRef.shape}')
            slices = (u0.shape[-3], u0.shape[-2], u0.shape[-1])
            # u0_50[:,:,index*inSize: (index+1)*inSize,:,:] = u0
            # uRef_50[:,:,index*inSize: (index+1)*inSize,:,:] = uRef  
            # u0_50[index*inSize: (index+1)*inSize,:,:,:,:] = u0
            # uRef_50[index*inSize: (index+1)*inSize,:,:,:,:] = uRef 
            u0_50[index,:,:,:,:] = u0
            uRef_50[index,:,:,:,:] = uRef  

    else:
        print(f'Running inference using {inSize} samples from 5 val dataset....')
        u0 = u0_50
        uRef = uRef_50
        slices = (u0.shape[-3], u0.shape[-2], u0.shape[-1])
        batchsize = 5
        out_index_start = time_list50[0]

    
    time = []
    compile_times = []
    uPred = np.zeros_like(uRef)
    print(f"Computing {tSteps}-Step prediction for {slices} with dt={model_dt}")
    input = u0
    single_step_error = []
    with torch.no_grad():
        for t in range(1,tSteps+1):
            start_inference = default_timer()
            uPred, compile_time = compile_timing(lambda: model(input))
            stop_inference = default_timer() - start_inference
            compile_times.append(compile_time)
            time.append(stop_inference)
            if torch.is_tensor(uPred):
                uPred = uPred.cpu().detach().numpy()
            if iVal < 20:
                if isinstance(start_idx, int):
                    uRef_single = u0_full[start_idx + int((t*model_dt)/dtData): end_idx + int((t*model_dt)/dtData)]
                    # print(f'{t}-step time: {time_org[start_idx + int((t*model_dt)/dtData): end_idx + int((t*model_dt)/dtData): 1]}')
                else:
                    out_index_single_step = start_idx + int((t*model_dt)/dtData)
                    uRef_single = u0_full[out_index_single_step]
                    # print(f'{t}-step time: {time_org[out_index_single_step]}')
                if ndim == 2:
                    uRef_single50[t-1, (index*inSize):(index+1)*inSize,:,:,:] = uRef_single
                else:
                    uRef_single = np.moveaxis(uRef_single, [0,2], [2,0])
                    # uRef_single50[t-1, :,:,(index*inSize):(index+1)*inSize,:,:] = uRef_single  
                    uRef_single50[t-1, index,:,:,:,:] = uRef_single   
                    # uRef_single50[t-1,(index*inSize):(index+1)*inSize, :,:,:,:] = uRef_single    
            else:
                uRef_single = uRef_single50[t-1]
            if ndim == 3:
                step_error_val = computeError(uPred, uRef_single, ndim, loss_axis).mean() #.mean(axis=2).mean(axis=1)[0]
            else:
                step_error_val = computeError(uPred, uRef_single, ndim, loss_axis).mean()
            single_step_error.append(step_error_val)
            input = uPred
    inferenceTime = np.round(sum(time),3)
    avg_inferenceTime = np.round(inferenceTime/len(time),3)
    print(f'Excluding initial compile inference timing: {compile_times[:2]}')
    compileTime = np.round(sum(compile_times[2:]),3)
    avg_compileTime = np.round(compileTime/(len(compile_times)-2),3)
    print(" -- done !")
    print(f'-- slices: {slices}')
    print(f'- -batchsize: {batchsize}')
    print(f' --shape of output: {uPred.shape}')
    print(f"-- Avg inference time on {device_name} (s) : {avg_inferenceTime}")
    print(f"-- Total inference time on {device_name} for {tSteps} iterations with dt of {model_dt} (s) : {inferenceTime}")
    print(f"-- Avg Compileinference time on {device_name} (s) : {avg_compileTime}")
    print(f"-- Total compile inference time on {device_name} for {tSteps} iterations with dt of {model_dt} (s) : {compileTime}")

    fig = plt.figure(f"{iVal}_step_error")
    singleerrorPlot = f"{iVal}_step_error.{imgExt}"
    plt.plot(np.arange(1,tSteps+1),single_step_error)
    ticks = [1, tSteps//3, 2*tSteps//3, tSteps]
    tick_label = [str(val*0.5) for val in ticks]
    plt.xticks(ticks, labels=tick_label)
    plt.ylabel('NRSSE', fontsize=14)
    plt.xlabel('Extrapolation in time / second', fontsize=14)
    plt.grid(True)
    fig.tight_layout()
    plt.savefig(f"{evalDir}/{singleerrorPlot}")
    plt.close()

    err = computeError(uPred, uRef, ndim, loss_axis)
    errId = computeError(u0, uRef, ndim, loss_axis)

    varNames = ["v_x", "v_z", "b", "p"]
    fig = plt.figure(f"D{slices}_error over time")
    for name, e, eId in zip(varNames, err.T, errId.T):
        p = plt.semilogy(e, '-', label=name, markevery=0.2)
        plt.semilogy(eId, '--', c=p[0].get_color())
    plt.legend()
    plt.grid(True)
    plt.xlabel("samples ordered with time")
    plt.ylabel("relative $L_2$ error")
    fig.set_size_inches(10, 5)
    plt.tight_layout()
    errorPlot = f"{iVal}_error_over_time.{imgExt}"
    plt.savefig(f"{evalDir}/{errorPlot}")
    plt.close()

    if ndim == 3:
        if loss_axis == 2:
            axis = 2
        else:
            axis = 0
        avgErr = err.mean(axis=axis)
        avgErrId = errId.mean(axis=axis)
    else:
        avgErr = err.mean(axis=0)
        avgErrId = errId.mean(axis=0)
    errors = pd.DataFrame(data={"model": avgErr.flatten(), "id": avgErrId.flatten()}, index=varNames)
    errors.loc["avg"] = errors.mean(axis=0)
    error_list.append(errors.mean(axis=0)[0])
    single_step_error_list.append(single_step_error)


    # -------------------------------------------------------------------------
    # -- Contour plots
    # -------------------------------------------------------------------------
    xGrid = dataset.infos["xGrid"][:]
    yGrid = dataset.infos["yGrid"][:]

    if ndim == 2:
        uI = u0[0, 2].T
        uM = uPred[0, 2].T
        uR = uRef[0, 2].T
    else:
        # uI = u0[0, 2, :,:,0].T
        # uM = uPred[0, 2,:,:,0].T
        # uR = uRef[0, 2, :,:,0].T
        uI = u0[0, 2, 0,:,:].T
        uM = uPred[0, 2,0,:,:].T
        uR = uRef[0, 2, 0,:,:].T

    contourPlotSol = f"{iVal}_contour_sol.{imgExt}"
    contourPlotStraat(
        uM, xGrid, yGrid, title="Model output",
        refField=uR, refTitle="Dedalus reference", time = time_org[out_index_start],
        saveFig=f"{evalDir}/{contourPlotSol}", refScales=True, closeFig=True)

    contourPlotErr = f"{iVal}_contour_err.{imgExt}"
    contourPlotStraat(
        np.abs(uM-uR), xGrid, yGrid, title="Absolute error",
        time = time_org[out_index_start],
        saveFig=f"{evalDir}/{contourPlotErr}", closeFig=True)
    
    contourPlot2(uM, xGrid, yGrid, title='', saveFig=f"{evalDir}/{iVal}_conotur_sol_paper.{imgExt}")
    contourPlot2(uR, xGrid, yGrid, title='', saveFig=f"{evalDir}/{iVal}_conotur_ref_paper.{imgExt}")

    # -------------------------------------------------------------------------
    # -- Averaged spectrum
    # -------------------------------------------------------------------------
    if ndim == 2:
        sxRef, szRef = computeMeanSpectrum(uRef)
        sxPred, szPred = computeMeanSpectrum(uPred)
    else:
        # sxRef, szRef = computeMeanSpectrum(uRef[:,:,:,:,0])
        # sxPred, szPred = computeMeanSpectrum(uPred[:,:,:,:,0])
        sxRef, szRef = computeMeanSpectrum(uRef[:,:,0,:,:])
        sxPred, szPred = computeMeanSpectrum(uPred[:,:,0,:,:])
    k = getModes(dataset.grid[0])

    plt.figure(f"{iVal}_spectrum")
    p = plt.loglog(k, sxRef.mean(axis=0), '--', label="sx (ref)")
    plt.loglog(k, sxPred.mean(axis=0), c=p[0].get_color(), label="sx (model)")

    p = plt.loglog(k, szRef.mean(axis=0), '--', label="sz (ref)")
    plt.loglog(k, szPred.mean(axis=0), c=p[0].get_color(), label="sz (model)")

    plt.legend()
    plt.grid()
    plt.ylabel("spectrum")
    plt.xlabel("wavenumber")
    plt.ylim(bottom=1e-10)
    plt.tight_layout()
    spectrumPlot = f"{iVal}_spectrum.{imgExt}"
    plt.savefig(f"{evalDir}/{spectrumPlot}")

    # -------------------------------------------------------------------------
    # -- Write slices evaluation in summary
    # -------------------------------------------------------------------------

    summary.write(TEMPLATE.format(
        iSimu=iVal,
        slices=slices,
        device=device_name,
        input_time=time_org[start_idx]+200,
        output_time=time_org[out_index_start]+200,
        errorPlot=errorPlot,
        errors=errors.to_markdown(floatfmt="1.1e"),
        avg_inferenceTime=avg_inferenceTime,
        avg_compileTime=avg_compileTime,
        tSteps=tSteps,
        dt=model_dt,
        compileTime=compileTime,
        inferenceTime=inferenceTime,
        contourPlotSol=contourPlotSol,
        contourPlotErr=contourPlotErr,
        spectrumPlot=spectrumPlot,
        ))

print(error_list)
avg_error = np.average(error_list)
avg_step_error = np.mean(single_step_error_list, axis=0)
# print(single_step_error_list)
fig = plt.figure(f"step_error")
steperrorPlot = f"step_error.{imgExt}"
plt.plot(np.arange(1,tSteps+1),avg_step_error)
ticks = [1, tSteps//3, 2*tSteps//3, tSteps]
tick_label = [str(val*0.5) for val in ticks]
plt.xticks(ticks, labels=tick_label)
plt.ylabel('NRSSE', fontsize=14)
plt.xlabel('Extrapolation in time / second', fontsize=14)
plt.grid(True)
fig.tight_layout()
plt.savefig(f"{evalDir}/{steperrorPlot}")
plt.close()

summary.write(f'Step Error: {single_step_error_list}\n')
summary.write(f'Average Error over all validation datasets = {avg_error}\n')
summary.close()
