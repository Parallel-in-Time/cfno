"""
Train a FNO3D model to map solution at T_in timesteps to next T timesteps
    
Usage:
    python fno3d.py \
            --run=<run_tracker> \
            --model_save_path=<save_dir> \
            --train_data_path=<train_data> \
            --val_data_path=<val_data> \
            --train_samples=<train_samples> \
            --val_samples=<validation_samples> \
            --input_timesteps=<T_in> \
            --output_timesteps=<T> \
            --start_index=<dedalus_start_index> \
            --stop_index=<dedalus_stop_index> \
            --time_slice=<dedalus_time_slice> \
            --dt=<dedalus_data_dt>
                 
    optional args:
        --single_data_path=<path to hdf5 file containing train, val and test data>
        --multi_step
        --load_checkpoint
        --checkpoint_path=<checkpoint_dir>
        --exit-signal-handler
        --exit-duration-in-mins
    
"""

import os
import sys
import h5py
import argparse
from pathlib import Path
import time
from tqdm import tqdm
from timeit import default_timer
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset, Dataset

from NeuralOperators.neural_operators.fnop.utils import CudaMemoryDebugger, format_tensor_size, LpLoss, UnitGaussianNormalizer, get_signal_handler, _set_signal_handler


def train(args):
    ## config
    train_samples = args.train_samples
    val_samples = args.val_samples
    
    T_in = args.input_timesteps
    T = args.output_timesteps
    start_index = args.start_index
    stop_index = args.stop_index
    timestep = args.time_slice
    dt = args.dt
    
    xStep = 1
    yStep = 1
    tStep = 1

    modes = 12
    width = 20

    epochs = 200
    batch_size = 5
    learning_rate = 0.00039
    weight_decay = 1e-05
    scheduler_step = 10.0
    scheduler_gamma = 0.98

    gridx = 4*256  # stacking [velx,velz,buoyancy,pressure]
    gridy = 64

    ## load data
    if args.single_data_path is not None:
        train_data_path = val_data_path = args.single_data_path
        train_reader = val_reader = h5py.File(train_data_path, mode="r")
    else:  
        train_data_path = args.train_data_path
        train_reader = h5py.File(train_data_path, mode="r")
        val_data_path = args.val_data_path
        val_reader = h5py.File(val_data_path, mode="r") 
    
    print('Starting data loading....')
    dataloader_time_start = default_timer()
    if args.multi_step:
        train_a, train_u = multi_data(train_reader,'train', start_index, stop_index, timestep, train_samples, T_in, T, xStep, yStep, tStep)
        val_a, val_u = multi_data(val_reader,'val', start_index, stop_index, timestep, val_samples, T_in, T, xStep, yStep, tStep)
    else:
        train_a = torch.tensor(train_reader['train'][:train_samples, ::xStep, ::yStep, start_index: start_index + (T_in*tStep): tStep], dtype=torch.float)
        train_u = torch.tensor(train_reader['train'][:train_samples, ::xStep, ::yStep, start_index + (T_in*tStep):  start_index + (T_in + T)*tStep: tStep], dtype=torch.float)

        val_a = torch.tensor(val_reader['val'][:val_samples, ::xStep, ::yStep, start_index: start_index + (T_in*tStep): tStep], dtype=torch.float)
        val_u = torch.tensor(val_reader['val'][:val_samples, ::xStep, ::yStep, start_index + (T_in*tStep):  start_index + (T_in + T)*tStep: tStep], dtype=torch.float)

    a_normalizer = UnitGaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    val_a = a_normalizer.encode(val_a)

    y_normalizer = UnitGaussianNormalizer(train_u)
    train_u = y_normalizer.encode(train_u)
    val_u = y_normalizer.encode(val_u)
    
    print(f"Train data:{train_u.shape}")
    print(f"Validation data:{val_u.shape}")
    
    nTrain = train_a.shape[0]
    nVal = val_a.shape[0]
    
    train_a = train_a.reshape(nTrain, gridx, gridy, 1, T_in).repeat([1,1,1,T,1])
    val_a = val_a.reshape(nVal, gridx, gridy, 1, T_in).repeat([1,1,1,T,1])
    
    print(f"Train data after reshaping:{train_u.shape}")
    print(f"Validation data after reshaping:{val_u.shape}")
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_a, val_u), batch_size=batch_size, shuffle=False)
    dataloader_time_stop = default_timer()
    print(f'Total time taken for dataloading (s): {dataloader_time_stop - dataloader_time_start}')
    
    iterations = epochs*(nTrain//batch_size)
    
    run = args.run
    fno_path = Path(f'{args.model_save_path}/rbc_fno3d_N{nTrain}_epoch{epochs}_m{modes}_w{width}_bs{batch_size}_dt{dt}_tin{T_in}_run{run}')
    fno_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(f'{fno_path}/checkpoint')
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=f"{fno_path}/tensorboard")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    torch.cuda.empty_cache()
    # memory = CudaMemoryDebugger(print_mem=True)

    ################################################################
    # training and evaluation
    ################################################################

    model3d = FNO3d(modes, modes, modes, width, T_in, T).to(device)
    n_params = model3d.print_size()
    # memory.print("After intialization")

    optimizer = torch.optim.Adam(model3d.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.AdamW(model3d.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    
    y_normalizer.to(device)
    myloss = LpLoss(size_average=False)
    start_epoch = 0

    print(f"fourier_modes: {modes}\n \
            layer_width: {width}\n \
            (Tin, Tout):{T_in, T}\n \
            batch_size: {batch_size}\n \
            optimizer: {optimizer}\n \
            lr_scheduler: {scheduler}\n \
            lr_scheduler_step: {scheduler_step}\n \
            lr_scheduler_gamma: {scheduler_gamma}\n\
            fno_path: {fno_path}")
    
    with open(f'{fno_path}/info.txt', 'a') as file:
        file.write("-------------------------------------------------\n")
        file.write(f"Model Card for FNO-3D with (x,z,t)\n")
        file.write("-------------------------------------------------\n")
        file.write(f"{n_params}\n")
        file.write("-------------------------------------------------\n")
        file.write(f"FNO config\n")
        file.write("-------------------------------------------------\n")
        file.write(f"Fourier modes:{modes}\n")
        file.write(f"Layer width:{width}\n")
        file.write(f"(nTrain, nVal): {nTrain, nVal}\n")
        file.write(f"Batchsize: {batch_size}\n")
        file.write(f"Optimizer: {optimizer}\n")
        file.write(f"LR scheduler: {scheduler}\n")
        file.write(f"LR scheduler step: {scheduler_step}\n")
        file.write(f"LR scheduler gamma: {scheduler_gamma}\n")
        file.write(f"Input timesteps given to FNO: {T_in}\n")
        file.write(f"Output timesteps given by FNO: {T}\n")
        file.write(f"Dedalus data dt: {dt}\n")
        file.write(f"Dedalus data start index: {start_index}\n")
        file.write(f"Dedalus data stop index: {stop_index}\n")
        file.write(f"Dedalus data slicing: {timestep}\n")
        file.write(f"(xStep, yStep, tStep): {xStep, yStep, tStep}\n")
        file.write(f"Grid(x,y): ({gridx, gridy})\n")
        file.write(f"Training data(input,output): {train_a.shape, train_u.shape}\n")
        file.write(f"FNO model path: {fno_path}\n")
        file.write("-------------------------------------------------\n")

    if args.load_checkpoint:
        checkpoint = torch.load(args.checkpoint_path)
        model3d.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_loss_l2 = checkpoint['l2_loss']
        start_loss_mse = checkpoint['mse_loss']
        start_val_l2 = checkpoint['val_loss']
        print(f"Continuing training from {checkpoint_path} at {start_epoch} with L2-loss {start_loss_l2},\
                MSE-loss {start_loss_mse} and Val-loss {start_val_l2}")
    
    print(f'Starting model training...')
    train_time_start = default_timer()   
    for epoch in range(start_epoch, epochs):
        with tqdm(unit="batch", disable=False) as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            
            t1 = default_timer()
            model3d.train()
            # memory.print("After model3d.train()")
    
            train_mse = 0
            train_l2 = 0
            
            for step, (xx, yy) in enumerate(train_loader):
                xx = xx.to(device)
                yy = yy.to(device)
                # memory.print("After loading first batch")

                pred = model3d(xx).view(batch_size, gridx, gridy, T)
                mse = nn.functional.mse_loss(pred, yy, reduction='mean')
                
                # print(f"{step} with encoding: y={yy.shape},x={xx.shape},pred={pred.shape}")
                
                yy = y_normalizer.decode(yy)
                pred = y_normalizer.decode(pred)
                
                # print(f"{step} decoded: y={yy.shape},x={xx.shape},pred={pred.shape}")
                
                l2 = myloss(pred.view(batch_size,-1), yy.view(batch_size, -1))
                train_l2 += l2.item()
                train_mse += mse.item()
                
                optimizer.zero_grad()
                l2.backward()
                optimizer.step()
                scheduler.step()
                
                grads = [param.grad.detach().flatten() for param in model3d.parameters()if param.grad is not None]
                grads_norm = torch.cat(grads).norm()
                writer.add_histogram("train/GradNormStep",grads_norm, step)
                
                # memory.print("After backwardpass")
                
                tepoch.set_postfix({'Batch': step + 1, 'Train l2 loss (in progress)': train_l2,\
                        'Train mse loss (in progress)': train_mse})
            
            # scheduler.step()
            train_l2_error = train_l2 / nTrain
            train_mse_error = train_mse / nTrain
            writer.add_scalar("train_loss/train_l2loss", train_l2_error, epoch)
            writer.add_scalar("train_loss/train_mseloss", train_mse_error, epoch)
            writer.add_scalar("train/GradNorm", grads_norm, epoch)
            
            val_l2 = 0
            model3d.eval()
            with torch.no_grad():
                for step, (xx,yy) in enumerate(val_loader):
                    xx = xx.to(device)
                    yy = yy.to(device)
                    # memory.print("after val first batch")
                    # pred = model3d(xx)
                    # print(f"Validation: {xx.shape}, {yy.shape}, {pred.shape}")
                    pred = model3d(xx).view(batch_size, gridx, gridy, T)
                    pred = y_normalizer.decode(pred)
                    yy = y_normalizer.decode(yy)
                    
                    val_l2 += myloss(pred.view(batch_size,-1), yy.view(batch_size, -1)).item()
                    

            val_l2_error = val_l2 / nVal
            writer.add_scalar("val_loss/val_l2loss", val_l2_error, epoch)
                
            t2 = default_timer()
            tepoch.set_postfix({ \
                'Epoch': epoch, \
                'Time per epoch (s)': (t2-t1), \
                'Train l2loss': train_l2_error ,\
                'Train mseloss': train_mse_error,\
                'Val l2loss': val_l2_error 
                })
            
        tepoch.close()
        
        if epoch > 0 and (epoch % 100 == 0 or epoch == epochs-1):
            with open(f'{fno_path}/info.txt', 'a') as file:
                file.write(f"Training loss at {epoch} epoch: {train_l2_error}\n")
                file.write(f"Validation loss at {epoch} epoch: {val_l2_error}\n")
            torch.save(
                {
                'epoch': epoch,
                'model_state_dict': model3d.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'l2_loss': train_l2_error,
                'mse_loss': train_mse_error,
                'val_loss': val_l2_error,
                }, f"{checkpoint_path}/model_checkpoint_{epoch}.pt")
            
        if args.exit_signal_handler:
            signal_handler = get_signal_handler()
            if any(signal_handler.signals_received()):
                with open(f'{fno_path}/info.txt', 'a') as file:
                    file.write(f"Training loss at {epoch} epoch: {train_l2_error}\n")
                    file.write(f"Validation loss at {epoch} epoch: {val_l2_error}\n")
                torch.save(
                   {
                    'epoch': epoch,
                    'model_state_dict': model3d.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'l2_loss': train_l2_error,
                    'mse_loss': train_mse_error,
                    'val_loss': val_l2_error,
                    }, f"{checkpoint_path}/model_checkpoint_{epoch}.pt")
                print('exiting program after receiving SIGTERM.')
                train_time_stop = default_timer()
                print(f'Total training+validation time (s): {train_time_stop - train_time_start}')
                sys.exit()
                
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_check = torch.tensor(
                [train_time > args.exit_duration_in_mins],
                dtype=torch.int, device=device)
            done = done_check.item()
            if done:
                with open(f'{fno_path}/info.txt', 'a') as file:
                    file.write(f"Training loss at {epoch} epoch: {train_l2_error}\n")
                    file.write(f"Validation loss at {epoch} epoch: {val_l2_error}\n")
                torch.save(
                    {
                    'epoch': epoch,
                    'model_state_dict': model3d.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'l2_loss': train_l2_error,
                    'mse_loss': train_mse_error,
                    'val_loss': val_l2_error,
                    }, f"{checkpoint_path}/model_checkpoint_{epoch}.pt")
                print('exiting program after {} minutes'.format(train_time))
                sys.exit()
                
    train_time_stop = default_timer()
    print(f'Exiting train()...')
    print(f'Total training+validation time (s): {train_time_stop - train_time_start}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FNO3D Training')
    parser.add_argument('--run', type=int, default=1,
                        help='training tracking number')
    parser.add_argument('--model_save_path', type=str,default=os.getcwd(),
                        help='path to which FNO model is saved')
    parser.add_argument('--single_data_path', type=str,default=None,
                        help='path to hdf5 file containing train, val and test data')
    parser.add_argument('--train_data_path', type=str,
                        help='path to train data hdf5 file')
    parser.add_argument('--val_data_path', type=str,
                        help='path to validation data hdf5 file')
    parser.add_argument('--train_samples', type=int, default=100,
                        help='Number of training samples')
    parser.add_argument('--val_samples', type=int, default=50,
                        help='Number of validation samples')
    parser.add_argument('--load_checkpoint', action="store_true",
                        help='load checkpoint')
    parser.add_argument('--checkpoint_path', type=str,
                        help='folder containing checkpoint')
    parser.add_argument('--multi_step', action="store_true",
                        help='take multiple step data')
    parser.add_argument('--input_timesteps', type=int, default=1,
                        help='number of input timesteps to FNO')
    parser.add_argument('--output_timesteps', type=int, default=1,
                        help='number of output timesteps to FNO')
    parser.add_argument('--start_index', type=int, 
                        help='starting time index for dedalus data')
    parser.add_argument('--stop_index', type=int, 
                        help='stopping time index for dedalus data')
    parser.add_argument('--time_slice', type=int, 
                        help='slicer for dedalus data')
    parser.add_argument('--dt', type=float, 
                        help='dedalus data dt')
    parser.add_argument('--exit-signal-handler', action='store_true',
                       help='Dynamically save the checkpoint and shutdown the '
                       'training if SIGTERM is received')
    parser.add_argument('--exit-duration-in-mins', type=int, default=None,
                       help='Exit the program after this many minutes.')
    args = parser.parse_args()

    if args.exit_signal_handler:
        _set_signal_handler()
    
    train(args)
    