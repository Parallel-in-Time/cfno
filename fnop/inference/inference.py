""""
Perform inference and plot results for 2D Rayleigh Benard Convection

Usage:
    python inference.py
        --config_file=<config_file> --model_checkpoint=<model_checkpoint>

"""

import os
import sys
sys.path.insert(1, os.getcwd())
import h5py
import argparse
from pathlib import Path
from timeit import default_timer
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from fnop.utils import CudaMemoryDebugger, DEFAULT_DEVICE, read_config
from fnop.data_procesing.data_utils import time_extract, state_extract
from fnop.models.fno2d import FNO2D
from fnop.models.fno3d import FNO3D
from fnop.utils import activation_selection

class FNOInference:
    """
    Inference for fourier neural operator
    """
    def __init__(self,
                 config,
                 checkpoint:str,
                 device:str=DEFAULT_DEVICE,
    ):
        """

        Args:
            config: YAML config file (either already loaded, or a path to it)
            checkpoint (str): torch model checkpoint path
            device (str, optional): cpu or cuda (default to cuda if available)
        """
        self.config = read_config(config)
        self.model_config = self.config.FNO
        self.data_config = self.config.data
        self.inference_config = self.config.inference
        self.checkpoint = checkpoint
        self.device = device
        self.memory = CudaMemoryDebugger(print_mem=True)

        self.nx = self.data_config.nx
        self.ny = self.data_config.ny
        if self.data_config.subdomain_process:
            self.nx = int(self.data_config.nx /self.data_config.subdomain_args.ndomain_x)
            self.ny = int(self.data_config.ny /self.data_config.subdomain_args.ndomain_y)
        self.nx_state = 4*self.nx
        self.T = self.inference_config.output_timesteps
        self.model = self.model_init()

    def model_init(self):
        non_linearity = nn.functional.gelu
        if self.model_config.non_linearity is not None:
            non_linearity = activation_selection(self.model_config.non_linearity)

        if self.config.dim == 'FNO3D':
            model = FNO3D(self.model_config.modes, self.model_config.modes,
                          self.model_config.modes,self.model_config.width,
                          self.model_config.T_in, self.T
                         ).to(self.device)
        else:
            model = FNO2D(self.model_config.modes,
                          self.model_config.modes,
                          self.model_config.lifting_width,
                          self.model_config.width,
                          self.model_config.projection_width,
                          self.model_config.n_layers,
                          self.model_config.T_in,
                          self.T,
                          non_linearity,
                        ).to(self.device)

        model_checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        if 'model_state_dict' in model_checkpoint.keys():
            model.load_state_dict(model_checkpoint['model_state_dict'])
        else:
            model.load_state_dict(model_checkpoint)

        return model

    def inference(self, input_data):
        """
        Perform inference

        Args:
            input_data (torch.tensor): FNO input tensor stack [velx, velz, buoyancy, pressure]

        Returns:
            pred (torch.tensor): FNO output stack [velx, velz, buoyancy, pressure]
        """
        input_data = torch.tensor(input_data, dtype=torch.float) if not torch.is_tensor(input_data) else input_data
        xx = input_data.to(self.device)
        xx_org = xx

        batch_size = self.inference_config.test_batch_size
        pred = torch.zeros([batch_size, self.nx_state, self.ny, self.T])

        self.model.eval()
        with torch.no_grad():
            if self.config.dim == 'FNO3D':
                pred = self.model(xx_org).view(batch_size, self.nx_state, self.ny, self.T)
                # pred = y_normalizer.decode(pred)
            else:
                for t in range(0, self.T, self.data_config.tStep):
                    im = self.model(xx)
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)

                    xx = torch.cat((xx[..., self.data_config.tStep:], im), dim=-1)

        return pred

    def predict(self, u0):
        v0 = self.numpy2Torch(u0)
        v1 = self.inference(v0)
        return self.torch2Numpy(v1)
    
    def __call__(self, u0):
        return self.predict(u0)

    def numpy2Torch(self, u:np.ndarray)->torch.tensor:
        nvar, nx, nz = u.shape
        self.nvar = nvar
        reshaped = u.reshape(1, nvar*nx, nz, 1)
        return torch.tensor(reshaped, dtype=torch.float)

    def torch2Numpy(self, u:torch.tensor)->np.ndarray:
        arr = u.cpu().detach().numpy()[0, ..., 0]  # extract only one time and first batch
        n1, nz = arr.shape
        nvar = self.nvar   # retrieved from previous numpy2Torch call
        nx = n1//nvar
        return arr.reshape(nvar, nx, nz)

    def output_formatter(self, outputs:torch.tensor)->np.ndarray:
        """
        Convert tensor to numpy array and extract states

        Args:
            outputs (torch.tensor): FNO model output of shape [batch, nx_state, ny, T]

        Returns:
            vx (np.ndarray): velocity x-componnent [nx, ny, T]
            vy (np.ndarray): velocity y-componnent [nx, ny, T]
            b (np.ndarray): buoyancy [nx, ny, T]
            p (np.ndarray):pressure [nx, ny, T]
        """
        outputs = outputs.cpu().detach().numpy()
        batch = 0
        vx, vy, b, p = state_extract(outputs[batch, :, :, :],
                                    self.nx,
                                    self.ny,
                                    self.T)

        return vx, vy, b, p

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
                                         self.nx,
                                         self.ny,
                                         self.T)

            vel = np.stack([vx, vy], axis=0)
            # print(f'Result:\nVelocity: {vel_pred}\nBuoyancy: {b_pred}\nPressure: {p_pred}')
            # print(f'vel: {vel_pred.shape}, b: {b_pred.shape}, p: {p_pred.shape}')
            file = f'{save_path}/inference_s{batch}.h5'
            v_shape = vel.shape  # [2, nx, ny, time]
            s_shape = p.shape    # [nx, ny, time]
            with h5py.File(file, "a") as data:
                data['scales/sim_time'] = time
                data['tasks/velocity'] = vel.reshape(v_shape[3], v_shape[0], v_shape[1], v_shape[2])
                data['tasks/buoyancy'] = b.reshape(s_shape[2],s_shape[0], s_shape[1])
                data['tasks/pressure'] = p.reshape(s_shape[2],s_shape[0], s_shape[1])

    def plot_cross_section( self,
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
                                         self.nx,
                                         self.ny,
                                         self.T)
            vx_pred, vy_pred, b_pred, p_pred = state_extract(predictions[batch, :, :, :],
                                                             self.nx,
                                                             self.ny,
                                                             self.T)
            for t in range(len(time)):
                row = 2
                col = 4
                xStep = 30
                yStep = 30
                x = np.arange(0,self.nx,xStep)
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

                fig.suptitle(f'RBC-2D with {self.nx}'+r'$\times$'+f'{self.ny} grid and Ra={rayleigh} and Pr={prandtl} using {self.config.dim}')
                ded_patch = Line2D([0], [0], label=f'Dedalus at t={np.round(time[t],4)}',marker='o', color='g')
                fno_patch = Line2D([0], [0], label=f'FNO at t={np.round(time[t],4)}',marker='o', linestyle='--', color='r')

                fig.legend(handles=[ded_patch, fno_patch], loc="upper right")
                # fig.tight_layout()
                fig.show()
                fig.savefig(f"{fno_path}/{self.config.dim}_NX{self.nx}_NY{self.ny}_{np.round(time[t],4)}_{batch}.png")

def main(model_checkpoint:str, config_file:str):
    #  Read the configuration
    config = read_config(config_file)
    nx_state = 4*config.data.nx
    device = torch.device(DEFAULT_DEVICE)
    print(f"Using {device}")
    if device == 'cuda':
        torch.cuda.empty_cache()
        # memory = CudaMemoryDebugger(print_mem=True)

    fno_path = Path(f'{config.inference.inference_save_path}/rbc_{config.dim}_m{config.FNO.modes}_w{config.FNO.width}_bs{config.data.batch_size}_dt{config.data.dt}_tin{config.FNO.T_in}_tout{config.inference.output_timesteps}_inference_{device}_run{config.run}')
    fno_path.mkdir(parents=True, exist_ok=True)

    start_index = config.inference.test_start_index
    dedalus_index =  config.inference.test_dedalus_index
    start_index_org =  dedalus_index + start_index
    time_in, time_out = time_extract(start_index_org, config.data.dt, config.FNO.T_in,
                                        config.inference.output_timesteps, config.data.tStep)
    print('Starting data loading....')
    dataloader_time_start = default_timer()
    test_reader=  h5py.File(config.inference.test_data_path, mode="r")

    inputs = None
    input_data = None
    outputs = None
    if config.data.subdomain_process:
        data = test_reader['val'][()]          # [time, ndomain_y*ndomain_x*samples, 4*subdomain_x, subdomain_y]
        inputs = data.reshape(data.shape[1], data.shape[2], data.shape[3], data.shape[0])
        input_data = torch.tensor(inputs[:config.inference.test_batch_size, ::config.data.xStep, ::config.data.yStep,
                                                    start_index:
                                                    start_index + (config.FNO.T_in*config.data.tStep):
                                                    config.data.tStep],
                                dtype=torch.float)
    else:
        input_data = torch.tensor(test_reader['test'][:config.inference.test_batch_size, ::config.data.xStep, ::config.data.yStep,
                                                    start_index:
                                                    start_index + (config.FNO.T_in*config.data.tStep):
                                                    config.data.tStep],
                                dtype=torch.float)

    if config.dim == 'FNO3D':
        input_data = input_data.reshape(config.inference.test_batch_size,
                                         nx_state, config.data.ny, 1,
                                         config.FNO.T_in).repeat([1,1,1,config.inference.output_timesteps,1])

    print(f'Input data: {input_data.shape}')

    if config.data.subdomain_process:
        outputs = torch.tensor(inputs[:config.inference.test_batch_size, ::config.data.xStep, ::config.data.yStep,
                                                start_index + (config.FNO.T_in*config.data.tStep):
                                                start_index + (config.FNO.T_in + config.inference.output_timesteps)*config.data.tStep:
                                                config.data.tStep],
                            dtype=torch.float)
    else:
        outputs = torch.tensor(test_reader['test'][:config.inference.test_batch_size, ::config.data.xStep, ::config.data.yStep,
                                                    start_index + (config.FNO.T_in*config.data.tStep):
                                                    start_index + (config.FNO.T_in + config.inference.output_timesteps)*config.data.tStep:
                                                    config.data.tStep],
                                dtype=torch.float)

    test_reader.close()
    dataloader_time_stop = default_timer()
    print(f'Total time taken for dataloading (s): {dataloader_time_stop - dataloader_time_start}')

    # print(f'input: {input_data.shape}, output: {outputs.shape}')

    print('Starting model inference...')
    model_inference = FNOInference(config, model_checkpoint, device=device)
    inference_time_start = default_timer()
    predictions = model_inference.inference(input_data)
    predictions_cpu = predictions.cpu()
    inference_time_stop = default_timer()
    # print(f'Inference shape: {predictions_cpu.shape}')
    print(f'Total time taken for model inference for {config.inference.output_timesteps} output {time_out} time steps with batchsize {config.inference.test_batch_size} and {config.FNO.T_in} input {time_in} time steps on {device} (s): {inference_time_stop - inference_time_start}')

    if config.inference.output_error:
        if config.dim == 'FNO3D':
            error = nn.functional.mse_loss(predictions_cpu, outputs, reduction='mean').item()
        else:
            error = nn.functional.mse_loss(predictions_cpu.reshape(config.inference.test_batch_size, -1), outputs.reshape(config.inference.test_batch_size, -1)).item()
        print(f'Mean Squared Error of FNO prediction = {error}')

    if config.inference.save_inference:
        model_inference.save_inference(fno_path, predictions_cpu.detach().numpy(), time_out)

    if config.inference.plot_cross_section:
        model_inference.plot_cross_section(fno_path, np.array(predictions_cpu),
                                         np.array(outputs), time_out,
                                         config.data.rayleigh_number,
                                         config.data.prandtl_number)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FNO Inference')
    parser.add_argument('--model_checkpoint', type=str,
                        help='model checkpoint path')
    parser.add_argument('--config_file', type=str,
                        help='config yaml file')
    args = parser.parse_args()
    main(args.model_checkpoint, args.config_file)
