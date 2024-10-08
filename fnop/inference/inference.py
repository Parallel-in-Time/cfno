""""
Perform inference and plot results for 2D Rayleigh Benard Convection

Usage:
    python inference.py 
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
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from fnop.utils import CudaMemoryDebugger
from fnop.data_procesing.data_utils import time_extract, state_extract
from fnop.models.fno2d import FNO2D
from fnop.models.fno3d import FNO3D
from fnop.utils import activation_selection

class FNOInference:
    """
    Inference for fourier neural operator
    """
    def __init__(self,
                 model:nn.Module,
                 dim: str,
                 dt:float,
                 gridx: int,
                 gridy: int,
                 T_in: int,
                 T:int,
                 xStep:int=1,
                 yStep:int=1,
                 tStep:int=1,
                 batch_size:int=1,
                 device:str='cpu',
    ): 
        """

        Args:
            model (nn.Module): FNO model
            dim (str): 'FNO2D' or 'FNO3D'
            dt (float): delta timestep
            gridx (int): size of x-grid
            gridy (int): size of y-grid
            T_in (int): number of input timesteps
            T (int): number of output timesteps
            xStep (int, optional): slicing for x-grid. Defaults to 1.
            yStep (int, optional): slicing for y-grid. Defaults to 1.
            tStep (int, optional): time slicing. Defaults to 1.
            batch_size (int, optional): inference batch size. Defaults to 1.
            device (str, optional): cpu or cuda
        """
        super().__init__()
        self.model = model
        self.dim = dim
        self.dt = dt
        self.gridx = gridx
        self.gridx_state = self.gridx*4
        self.gridy = gridy
        self.T_in = T_in
        self.T = T
        self.xStep = xStep
        self.yStep = yStep
        self.tStep = tStep
        self.batch_size = batch_size
        self.device = device
        self.memory = CudaMemoryDebugger(print_mem=True)

    @staticmethod
    def fromFiles(checkpoint_file:str, config_file:str):
        pass # return a FNOInference object.
    
    def inference(self, input_data):
        """
        Perform inference

        Args:
            input_data (torch.tensor): FNO input tensor stack [velx, velz, buoyancy, pressure]
            
        Returns:
            pred (torch.tensor): FNO output stack [velx, velz, buoyancy, pressure]
        """
        
        pred = torch.zeros([self.batch_size, self.gridx_state, self.gridy, self.T])
        xx = input_data.to(self.device)
        xx_org = xx
        self.model.eval()
        with torch.no_grad():
            if self.dim == 'FNO3D':
                pred = self.model(xx_org).view(self.batch_size, self.gridx_state, self.gridy, self.T)
                # pred = y_normalizer.decode(pred)
            else:
                for t in range(0, self.T, self.tStep):
                    im = self.model(xx)
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
            
                    xx = torch.cat((xx[..., self.tStep:], im), dim=-1)
            
        return pred
    
    def predict(self, u0:np.ndarray)->np.ndarray:
        pass # interface to inference with conversion numpy - tensor in between ...
        
    def save_inference(self,
                       save_path:str,
                       predictions:np.ndarray,
                       time: list,
    ):
        """
        Save inference result to hdf5 file

        Args:
            save_path (str): root path to save inference hdf5 file
            predictions (np.ndarray): FNO model output
            time (list): simulation time
        """
        for batch in range(predictions.shape[0]):
            vx, vy, b, p = state_extract(predictions[batch, :, :, :], 
                                         self.gridx,
                                         self.gridy,
                                         self.T)
        
            vel = np.stack([vx, vy], axis=0)
            # print(f'Result:\nVelocity: {vel_pred}\nBuoyancy: {b_pred}\nPressure: {p_pred}')
            # print(f'vel: {vel_pred.shape}, b: {b_pred.shape}, p: {p_pred.shape}')
            file = f'{save_path}/inference_s{batch}.h5'
            v_shape = vel.shape  # [dim, gridx, gridy, time]
            s_shape = p.shape    # [gridx, gridy, time]
            with h5py.File(file, "a") as data:
                data['scales/sim_time'] = time
                data['tasks/velocity'] = vel.reshape(v_shape[3], v_shape[0], v_shape[1], v_shape[2])
                data['tasks/buoyancy'] = b.reshape(s_shape[2],s_shape[0], s_shape[1])
                data['tasks/pressure'] = p.reshape(s_shape[2],s_shape[0], s_shape[1])
                
    def plot_cross_section(self,
                        fno_path:str,
                        predictions: np.ndarray,
                        output: np.ndarray,
                        time:list,
                        rayleigh:float,
                        prandtl:float,
    ):
        """
        Plotting cross-sections of velocity, buoyancy and pressure data 
        on grid with error bars 

        Args:
            fno_path (str): path to store plots
            predictions (np.ndarray): FNO model output
            output (np.ndarray): Dedalus output
            time (list): list of output simulation times
            rayleigh (float): Rayleigh Number
            prandtl (float): Prandtl number 
            

        """
        for batch in range(predictions.shape[0]):
            vx, vy, b, p = state_extract(output[batch, :, :, :], 
                                         self.gridx,
                                         self.gridy,
                                         self.T)
            vx_pred, vy_pred, b_pred, p_pred = state_extract(predictions[batch, :, :, :], 
                                                             self.gridx,
                                                             self.gridy,
                                                             self.T)
            for t in range(len(time)):
                row = 2
                col = 4
                xStep = 30
                yStep = 30
                x = np.arange(0,self.gridx,xStep)
                fig, ax = plt.subplots(nrows=row, 
                                    ncols=col,
                                    figsize=(16, 12),
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
                ax1.plot(x,vx[::xStep,::yStep,t],color='g',marker ='o',label="ded-vx")
                ax1.plot(x,vx_pred[::xStep,::yStep,t],color='r',marker ='o',ls='--',label="fno-vx")
                # ax1.set_ylabel("Y grid")
                ax1.grid()

                ax2.set_title(fr'Velocity: $u(z)$ ')
                ax2.plot(x,vy[::xStep,::yStep,t],marker ='o',color='g',label="ded-vy")
                ax2.plot(x,vy_pred[::xStep,::yStep,t],marker ='o',color='r',linestyle='--',label="fno-vy")
                # ax2.set_ylabel("Y grid")
                ax2.grid()

                ax3.set_title(fr'Pressure: $p(x,z)$')
                ax3.plot(x,p[::xStep,::yStep,t],marker ='o',color='g',label="ded-p")
                ax3.plot(x,p_pred[::xStep,::yStep,t],marker ='o',color='r',linestyle='--',label="fno-p")
                # ax3.set_ylabel("Y grid")
                ax3.grid()

                ax4.set_title(fr'Buoyancy: $b(x,z)$')
                ax4.plot(x,b[::xStep,::yStep,t],marker ='o',color='g',label="ded-b")
                ax4.plot(x,b_pred[::xStep,::yStep,t],marker ='o',linestyle='--',color='r',label="fno-b")
                # ax4.set_ylabel("Y grid")
                ax4.grid()
                
                ax5.errorbar(x, np.average(vx[::xStep,::yStep,t],axis=1), yerr=np.average(np.abs(vx_pred[::xStep,::yStep,t]-vx[::xStep,::yStep,t]), axis=1), marker='o',color='purple',capsize=3, markersize=6,linestyle='none')
                ax5.set_ylabel(r"$\overline{|vx_{ded}-vx_{fno}|}_{z}$")
                ax5.set_xlabel("X Grid")

                ax6.errorbar(x, np.average(vy[::xStep,::yStep,t],axis=1), yerr=np.average(np.abs(vy_pred[::xStep,::yStep,t]-vy[::xStep,::yStep,t]), axis=1), marker='o',color='purple',capsize=3, markersize=6,linestyle='none')
                ax6.set_ylabel(r"$\overline{|vz_{ded}-vz_{fno}|}_{z}$")
                ax6.set_xlabel("X Grid")

                ax7.errorbar(x, np.average(p[::xStep,::yStep,t],axis=1), yerr=np.average(np.abs(p_pred[::xStep,::yStep,t]-p[::xStep,::yStep,t]), axis=1), marker='o',color='purple',capsize=3, markersize=6,linestyle='none')
                ax7.set_ylabel(r"$\overline{|p_{ded}-p_{fno}|}_{z}$")
                ax7.set_xlabel("X Grid")

                ax8.errorbar(x, np.average(b[::xStep,::yStep,t],axis=1), yerr=np.average(np.abs(b_pred[::xStep,::yStep,t]-b[::xStep,::yStep,t]), axis=1), marker='o',color='purple',capsize=3, markersize=6,linestyle='none')
                ax8.set_ylabel(r"$\overline{|b_{ded}-b_{fno}|}_{z}$")
                ax8.set_xlabel("X Grid")

                fig.suptitle(f'RBC-2D with {self.gridx}'+r'$\times$'+f'{self.gridy} grid and Ra={rayleigh} and Pr={prandtl} using {self.dim}')  
                ded_patch = Line2D([0], [0], label=f'Dedalus at t={np.round(time[t],4)}',marker='o', color='g')
                fno_patch = Line2D([0], [0], label=f'FNO at t={np.round(time[t],4)}',marker='o', linestyle='--', color='r')
            
                fig.legend(handles=[ded_patch, fno_patch], loc="upper right")
                # fig.tight_layout()
                fig.show()
                fig.savefig(f"{fno_path}/{self.dim}_NX{self.gridx}_NY{self.gridy}_{np.round(time[t],4)}_{batch}.png")
        
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
    inference_config = config.inference
    gridx_state = 4*data_config.gridx
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    if device == 'cuda':
        torch.cuda.empty_cache()
        # memory = CudaMemoryDebugger(print_mem=True)

    fno_path = Path(f'{inference_config.inference_save_path}/rbc_{config.dim}_m{model_config.modes}_w{model_config.width}_bs{data_config.batch_size}_dt{data_config.dt}_tin{model_config.T_in}_tout{inference_config.output_timesteps}_inference_{device}_run{config.run}')
    fno_path.mkdir(parents=True, exist_ok=True)
    non_linearity = nn.functional.gelu
    if model_config.non_linearity is not None:
        non_linearity = activation_selection(model_config.non_linearity)
    
    if config.dim == 'FNO3D':
        model = FNO3D(model_config.modes, model_config.modes, 
                      model_config.modes,model_config.width, 
                      model_config.T_in, inference_config.output_timesteps
                      ).to(device)
    else:
        model = FNO2D(model_config.modes, 
                  model_config.modes, 
                  model_config.lifting_width,
                  model_config.width, 
                  model_config.projection_width,
                  model_config.n_layers,
                  model_config.T_in,
                  inference_config.output_timesteps,
                  non_linearity,
                ).to(device)

    model_checkpoint = torch.load(inference_config.model_checkpoint, map_location=lambda storage, loc: storage)
    if 'model_state_dict' in model_checkpoint.keys():
        model.load_state_dict(model_checkpoint['model_state_dict'])
    else:
        model.load_state_dict(model_checkpoint)
        
    model_inference = FNOInference(model=model,
                             dim=config.dim,
                             dt=data_config.dt,
                             gridx=data_config.gridx,
                             gridy=data_config.gridy,
                             T_in=model_config.T_in,
                             T=inference_config.output_timesteps,
                             xStep=data_config.xStep,
                             yStep=data_config.yStep,
                             tStep=data_config.tStep,
                             batch_size=inference_config.test_batch_size,
                             device=device)

    start_index = inference_config.test_start_index
    dedalus_index =  inference_config.test_dedalus_index
    start_index_org =  dedalus_index + start_index
    time_in, time_out = time_extract(start_index_org, data_config.dt, model_config.T_in, 
                                        inference_config.output_timesteps, data_config.tStep)
    print('Starting data loading....')
    dataloader_time_start = default_timer()
    test_reader=  h5py.File(inference_config.test_data_path, mode="r")
    input_data = torch.tensor(test_reader['test'][:inference_config.test_batch_size, ::data_config.xStep, ::data_config.yStep, 
                                                    start_index: 
                                                    start_index + (model_config.T_in*data_config.tStep): 
                                                    data_config.tStep],
                                dtype=torch.float)
    print(input_data.shape)
    if config.dim == 'FNO3D':
        input_data = input_data.reshape(inference_config.test_batch_size,
                                         gridx_state, data_config.gridy, 1,
                                         model_config.T_in).repeat([1,1,1,inference_config.output_timesteps,1])
         
    outputs = torch.tensor(test_reader['test'][:inference_config.test_batch_size, ::data_config.xStep, ::data_config.yStep, 
                                                start_index + (model_config.T_in*data_config.tStep): 
                                                start_index + (model_config.T_in + inference_config.output_timesteps)*data_config.tStep:
                                                data_config.tStep],
                            dtype=torch.float)
        
    dataloader_time_stop = default_timer()
    print(f'Total time taken for dataloading (s): {dataloader_time_stop - dataloader_time_start}')
    
    # print(f'input: {input_data.shape}, output: {outputs.shape}')
    print('Starting model inference...')
    inference_time_start = default_timer()
    predictions = model_inference.inference(input_data)
    predictions_cpu = predictions.cpu()
    inference_time_stop = default_timer()
    # print(f'Inference shape: {predictions_cpu.shape}')
    print(f'Total time taken for model inference for {inference_config.output_timesteps} output {time_out} steps\nwith batchsize {inference_config.test_batch_size} and {model_config.T_in} input {time_in} steps\non {device} (s): {inference_time_stop - inference_time_start}')
    
    if inference_config.output_error:
        if config.dim == 'FNO3D':
            error = nn.functional.mse_loss(predictions_cpu, outputs, reduction='mean').item()
        else:
            error = nn.functional.mse_loss(predictions_cpu.reshape(inference_config.test_batch_size, -1), outputs.reshape(inference_config.test_batch_size, -1)).item()
        print(f'Mean Squared Error of FNO prediction = {error}')
    
    if inference_config.save_inference:
        model_inference.save_inference(fno_path, np.array(predictions_cpu), time_out)
         
    if inference_config.plot_cross_section:
        model_inference.plot_cross_section(fno_path, np.array(predictions_cpu),
                                         np.array(outputs), time_out,
                                         data_config.rayleigh_number,
                                         data_config.prandtl_number)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FNO Inference')
    parser.add_argument('--config_file', type=str,
                        help='config yaml file')
    args = parser.parse_args()
    
    main(args.config_file)
    