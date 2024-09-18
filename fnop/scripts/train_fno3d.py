"""
Train a FNO3D model to map solution at T_in timesteps to next T timesteps
    
Usage:
    python fno3d.py 
     --config=<config_file>
    
"""

import os
import sys
sys.path.insert(1, os.getcwd())
import h5py
import argparse
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from pathlib import Path
from timeit import default_timer
import torch
from torch.utils.tensorboard import SummaryWriter
from fnop.utils import _set_signal_handler, CudaMemoryDebugger
from fnop.data_procesing.data_loader import FNODataLoader
from fnop.models.fno3d import FNO3D
from fnop.losses.data_loss import LpLoss
from fnop.training.trainer import Trainer

_GLOBAL_SIGNAL_HANDLER = None

def main(config_file:str):
    # Read the configuration
    pipe = ConfigPipeline(
        [
            YamlConfig(config_file),
            ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        ]
    )
    config = pipe.read_conf()
    model_config = config.FNO
    data_config = config.data
    opt_config = config.opt
  
    if config.exit_signal_handler is not None:
        _set_signal_handler()
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    torch.cuda.empty_cache()
    memory = CudaMemoryDebugger(print_mem=True)

    print('Starting data loading....')
    dataloader_time_start = default_timer()
    loader = FNODataLoader( batch_size=data_config.batch_size,
                            gridx= data_config.gridx, gridy=data_config.gridy,
                            dt=data_config.dt, dim=config.dim, xStep=data_config.xStep, yStep=data_config.yStep, 
                            tStep=data_config.tStep, start_index=data_config.start_index, 
                            stop_index=data_config.stop_index, timestep=data_config.timestep,
                            T_in=model_config.T_in, T=model_config.T
                        )
    
    if data_config.data_path is not None:
        train_reader = val_reader =  h5py.File(data_config.data_path, mode="r")
    else:  
        train_reader = h5py.File(data_config.train_data_path, mode="r")
        val_reader = h5py.File(data_config.val_data_path, mode="r") 

    train_loader = loader.data_loader('train', data_config.train_samples, train_reader)
    val_loader = loader.data_loader('val', data_config.val_samples, val_reader)
    dataloader_time_stop = default_timer()
    print(f'Total time taken for dataloading (s): {dataloader_time_stop - dataloader_time_start}')

    fno_path = Path(f'{config.save_path}/rbc_fno3d_N{data_config.train_samples}_epoch{opt_config.epochs}_m{model_config.modes}_w{model_config.width}_bs{data_config.batch_size}_dt{data_config.dt}_tin{model_config.T_in}_{device}_run{config.run}')
    fno_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = Path(f'{fno_path}/checkpoint')
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    tensorboard_writer = SummaryWriter(log_dir=f"{fno_path}/tensorboard")

    ################################################################
    # training and evaluation
    ################################################################

    model = FNO3D(model_config.modes, model_config.modes, model_config.modes,
                  model_config.width, model_config.T_in, model_config.T).to(device)
    memory.print("after intialization")

    optimizer = torch.optim.Adam(model.parameters(), lr=opt_config.learning_rate, weight_decay=opt_config.weight_decay)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=opt_config.learning_rate, weight_decay=opt_config.weight_decay)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_config.T_max)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt_config.scheduler_step, gamma=opt_config.scheduler_gamma)

    loss = LpLoss(size_average=False)

    if config.verbose:
        with open(f'{fno_path}/info.txt', 'a') as file:
            file.write("-------------------------------------------------\n")
            file.write(f"Model Card for FNO-3D with (x,y,t)\n")
            file.write("-------------------------------------------------\n")
            file.write(f"{model.print_size()}\n")
            file.write("-------------------------------------------------\n")
            file.write(f"FNO config\n")
            file.write("-------------------------------------------------\n")
            file.write(f"Fourier modes:{model_config.modes}\n")
            file.write(f"Layer width:{model_config.width}\n")
            file.write(f"(nTrain, nVal, nTest): {data_config.train_samples, data_config.val_samples, data_config.test_samples}\n")
            file.write(f"Batchsize: {data_config.batch_size}\n")
            file.write(f"Optimizer: {optimizer}\n")
            file.write(f"LR scheduler: {scheduler}\n")
            file.write(f"LR scheduler step: {opt_config.scheduler_step}\n")
            file.write(f"LR scheduler gamma: {opt_config.scheduler_gamma}\n")
            file.write(f"Input timesteps given to FNO: {model_config.T_in}\n")
            file.write(f"Output timesteps given by FNO: {model_config.T}\n")
            file.write(f"Dedalus data dt: {data_config.dt}\n")
            file.write(f"Dedalus data start index: {data_config.start_index}\n")
            file.write(f"Dedalus data stop index: {data_config.stop_index}\n")
            file.write(f"Dedalus data slicing: {data_config.timestep}\n")
            file.write(f"(xStep, yStep, tStep): {data_config.xStep, data_config.yStep, data_config.tStep}\n")
            file.write(f"Grid(x,y): ({data_config.gridx, data_config.gridy})\n")
            file.write(f"FNO model path: {fno_path}\n")
            file.write("-------------------------------------------------\n")

    trainer = Trainer(
        model=model,
        dim=config.dim,
        epochs= opt_config.epochs,
        dt=data_config.dt,
        gridx=data_config.gridx,
        gridy=data_config.gridy,
        T_in=model_config.T_in,
        T=model_config.T,
        xStep=data_config.xStep,
        yStep=data_config.yStep,
        tStep=data_config.tStep,
        exit_signal_handler=config.exit_signal_handler,
        exit_duration_in_mins=config.exit_duration_in_mins,
        device=device
    )
    
    trainer.train(
        save_path=fno_path,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=data_config.batch_size,
        training_loss=loss,
        val_loss=loss,
        nTrain=data_config.train_samples,
        nVal=data_config.val_samples,
        tensorboard_writer=tensorboard_writer,
        resume_from_checkpoint=config.resume_from_checkpoint
    )
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FNO3D Training')
    parser.add_argument('--config_file', type=str,
                        help='FNO2D config yaml file')
    args = parser.parse_args()
    
    main(args.config_file)
    

    