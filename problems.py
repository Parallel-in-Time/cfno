import random

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from fourier_operator import FourierFeatures, FNO2d

from torch.utils.data import Dataset

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# NOTE:
#All the training sets should be in the folder: data/

  
        
def default_param(network_properties):
    
    if "modes" not in network_properties:
        network_properties["modes"] = 16
    
    if "width" not in network_properties:
        network_properties["width"] = 32
    
    if "n_layers" not in network_properties:
        network_properties["n_layers"] = 4
        
    if "proj_scale" not in network_properties:
        network_properties["proj_scale"] = 32

    if "padding" not in network_properties:
        network_properties["padding"] = 0
    
    if "include_grid" not in network_properties:
        network_properties["include_grid"] = 1
    
    if "FourierF" not in network_properties:
        network_properties["FourierF"] = 0
    
    return network_properties


def default_train_params(training_properties):
    if "learning_rate" not in training_properties:
        training_properties["learning_rate"] = 0.001
        
    if "weight_deacy" not in training_properties:
        training_properties["weight_decay"] = 1e-8
        
    if "scheduler_step" not in training_properties:
        training_properties["scheduler_step"] = 0.97
        
    if "scheduler_gamma" not in training_properties:
        training_properties["scheduler_gamma"] = 10
        
    if "epochs" not in training_properties:
        training_properties["epochs"] = 1000
        
    if "batch_size" not in training_properties:
        training_properties["batch_size"] = 16
        
    if "exp" not in training_properties:
        training_properties["exp"] = 1
        
    if "training_samples" not in training_properties:
        training_properties["training_samples"] = 250
        
    return training_properties


def RBC_param(network_properties):
    
    if "modes1" not in network_properties:
        network_properties["modes1"] = 32
        
    if "modes2" not in network_properties:
        network_properties["modes2"] = 32
    
    if "width" not in network_properties:
        network_properties["width"] = 128
    
    if "n_layers" not in network_properties:
        network_properties["n_layers"] = 2
        
    if "proj_scale" not in network_properties:
        network_properties["proj_scale"] = 32

    if "padding" not in network_properties:
        network_properties["padding"] = 0
    
    if "include_grid" not in network_properties:
        network_properties["include_grid"] = 0
    
    if "FourierF" not in network_properties:
        network_properties["FourierF"] = 0
        
    if "activation" not in network_properties:
        network_properties["activation"] = 'ReLU'
    
    return network_properties


def RBC_train_param(training_properties):
    if "learning_rate" not in training_properties:
        training_properties["learning_rate"] = 0.00039
        
    if "weight_deacy" not in training_properties:
        training_properties["weight_decay"] = 1e-05
        
    if "scheduler_step" not in training_properties:
        training_properties["scheduler_step"] = 10
        
    if "scheduler_gamma" not in training_properties:
        training_properties["scheduler_gamma"] = 0.98
        
    if "epochs" not in training_properties:
        training_properties["epochs"] = 100
        
    if "batch_size" not in training_properties:
        training_properties["batch_size"] = 30
        
    if "exp" not in training_properties:
        training_properties["exp"] = 1
        
    if "training_samples" not in training_properties:
        training_properties["training_samples"] = 99
        
    if "start_t" not in training_properties:
        training_properties["start_t"] = 100
        
    if "end_t" not in training_properties:
        training_properties["end_t"] = 198
    
    return training_properties    



# Wave data:
# From 0 to 512 : training samples (512)
# From 1024 to 1024 + 128 : validation samples (128)
# From 1024 + 128 to 1024 + 128 + 256 : test samples (256)
# Out-of-distribution testing samples: 0 to 256 (256)

class WaveEquationDataset(Dataset):
    def __init__(self, task="training", nf=0, training_samples = 1024, t = 5, s = 64, in_dist = True):
        
        # Data file:       
        if in_dist:
            self.file_data = "data/WaveData_64x64_IN.h5"
        else:
            self.file_data = "data/WaveData_64x64_OUT.h5"

        self.reader = h5py.File(self.file_data, 'r')
        
        # Load normaliation constants:
        self.min_data = self.reader['min_u0'][()]
        self.max_data = self.reader['max_u0'][()]
        self.min_model = self.reader['min_u'][()]
        self.max_model = self.reader['max_u'][()]
        
        # Time
        self.t = t
                        
        if task == "training":
            self.length = training_samples
            self.start = 0
        elif task == "validation":
            self.length = 128
            self.start = 1024
        elif task == "test":
            if in_dist:
                self.length = 256
                self.start = 1024 + 128
            else:
                self.length = 256
                self.start = 0
                
        # Grid size
        self.s = s
        
        #Fourier modes 
        self.N_Fourier_F = nf
        
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)+"_t_"+str(self.t)]["input"][:]).type(torch.float32).reshape(1, self.s, self.s)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)+"_t_"+str(self.t)]["output"][:]).type(torch.float32).reshape(1, self.s, self.s)
        # normalising data 
        inputs = (inputs - self.min_data)/(self.max_data - self.min_data)
        labels = (labels - self.min_model)/(self.max_model - self.min_model)

        if self.N_Fourier_F > 0:
            grid = self.get_grid() # [s, s, 2]
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            # B shape torch.Size([N_Fourier_F, 2])
            # x_proj torch.Size([s, s, N_Fourier_F])
            ff_grid = FF(grid) # [s, s, 2*N_Fourier_F]
            ff_grid = ff_grid.permute(2, 0, 1) #[s, s, 2*N_Fourier_F] ---> [2*N_Fourier_F,s,s]
            inputs = torch.cat((inputs, ff_grid), 0) #([1,s,s],[2*N_Fourier,s,s]) ---> [2*N_Fourier_F+1,s,s]
            

        return inputs.permute(1, 2, 0), labels.permute(1, 2, 0) # [2*N_Fourier_F+1,s,s], [1,s,s] ---> [s,s,2*N_Fourier_F+1], [s,s,1]

    def get_grid(self):
        x = torch.linspace(0, 1, self.s) # [s]
        y = torch.linspace(0, 1, self.s) # [s]

        x_grid, y_grid = torch.meshgrid(x, y) # [s,s]

        x_grid = x_grid.unsqueeze(-1)  # [s,s,1]
        y_grid = y_grid.unsqueeze(-1)  # [s,s,1]
        grid = torch.cat((x_grid, y_grid), -1)  # [s,s,2]
        return grid


class WaveEquation:
    def __init__(self, network_properties, device, batch_size, training_samples = 1024, s = 64, in_dist = True):
        
        network_properties = default_param(network_properties)
        self.N_Fourier_F = network_properties["FourierF"]
    
        
        self.model = FNO2d(fno_architecture = network_properties, 
                            in_channels = 1 + 2 * self.N_Fourier_F, 
                            out_channels = 1, 
                            device=device) 
        num_workers = 8
        
        self.train_loader = DataLoader(WaveEquationDataset("training", self.N_Fourier_F, training_samples, 5, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(WaveEquationDataset("validation", self.N_Fourier_F, training_samples, 5, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(WaveEquationDataset("test", self.N_Fourier_F, training_samples, 5, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        # self.model = FNO1d(fno_architecture = network_properties,
                    # in_channels=2,
                    # out_channels=1,
                    # padding_frac=1/4,
                    # device=device)
        
        
        
# Darcy Flow data
#   From 0 to 256 : training samples (256)
#   From 256 to 256 + 128 : validation samples (128)
#   From 256 + 128 to 256 + 128 + 128 : test samples (128)
#   Out-of-distribution testing samples: 0 to 128 (128)

class DarcyDataset(Dataset):
    def __init__(self, task="training", nf=0, training_samples=256, s=64, insample=True):

        if insample:
            self.file_data = "data/Darcy_64x64_IN.h5"
        else:
            self.file_data = "data/Darcy_64x64_IN.h5"
        
        
        self.reader = h5py.File(self.file_data, 'r')

        self.min_data = self.reader['min_inp'][()]
        self.max_data = self.reader['max_inp'][()]
        self.min_model = self.reader['min_out'][()]
        self.max_model = self.reader['max_out'][()]
                
        if task == "training":
            self.length = training_samples
            self.start = 0
        elif task == "validation":
            self.length = 128
            self.start = training_samples
        elif task == "testing":
            if insample:
                self.length = 128
                self.start = training_samples + 128
            else:
                self.length = 128
                self.start = 0

        self.N_Fourier_F = nf
        self.s = s

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.reader['sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, self.s, self.s)
        labels = torch.from_numpy(self.reader['sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, self.s, self.s)
        # normalisation
        inputs = (inputs - self.min_data) / (self.max_data - self.min_data)
        labels = (labels - self.min_model) / (self.max_model - self.min_model)

        if self.N_Fourier_F > 0:
            grid = self.get_grid()  # [s, s, 2*N_Fourier_F]
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
             # B shape torch.Size([N_Fourier_F, 2])
            # x_proj torch.Size([s, s, N_Fourier_F])
            ff_grid = FF(grid) # [s, s, 2*N_Fourier_F]
            ff_grid = ff_grid.permute(2, 0, 1)  #[s, s, 2*N_Fourier_F] ---> [2*N_Fourier_F,s,s]
            inputs = torch.cat((inputs, ff_grid), 0) #([1,s,s],[2*N_Fourier,s,s]) ---> [2*N_Fourier_F+1,s,s]

        return inputs.permute(1, 2, 0), labels.permute(1, 2, 0) # [2*N_Fourier_F+1,s,s], [1,s,s] ---> [s,s,2*N_Fourier_F+1], [s,s,1]

    def get_grid(self):
        x = torch.linspace(0, 1, s) # [s]
        y = torch.linspace(0, 1, s) # [s]

        x_grid, y_grid = torch.meshgrid(x, y) # [s,s]

        x_grid = x_grid.unsqueeze(-1) # [s,s,1]
        y_grid = y_grid.unsqueeze(-1)  # [s,s,1]
        grid = torch.cat((x_grid, y_grid), -1)  # [s,s,2]
  
        return grid

class Darcy:
    def __init__(self, network_properties, device, batch_size, training_samples = 512,  s = 64, in_dist = True):
        
        network_properties = default_param(network_properties)
        self.N_Fourier_F = network_properties["FourierF"]
        
        self.model = FNO2d(fno_architecture = network_properties, 
                            in_channels = 1 + 2 * self.N_Fourier_F, 
                            out_channels = 1, 
                            device=device) 
        
        # self.model = FNO1d(fno_architecture = network_properties,
                           # in_channels=2,
                           # out_channels=1,
                           # padding_frac=1/4,
                           # device=device)
                           

        num_workers = 8

        self.train_loader = DataLoader(DarcyDataset("training", self.N_Fourier_F, training_samples), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(DarcyDataset("validation", self.N_Fourier_F, training_samples), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(DarcyDataset("testing", self.N_Fourier_F, training_samples, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
  
  
# RBC-2D Data
# Train: 250 samples
# Validation: 100 samples
# Test: 100 samples

class RBCDataset2D(Dataset):
    def __init__(self, task="training", nf=0, samples = 250, start_t = 0, end_t = 10, sx = 256, sz = 64):
    
        
        # Data file: 
        if task == "training":
            self.file_data = "/p/project/cexalab/john2/NeuralOperators/RayleighBernardConvection/RBC2D_NX64_NZ64_TF50_Pr1_Ra10e4/RBC2D_NX64_NZ64_TF50_Pr1_Ra10e4_train.h5"
            self.length = samples
        elif task == "validation":
            self.file_data = "/p/project/cexalab/john2/NeuralOperators/RayleighBernardConvection/RBC2D_NX64_NZ64_TF50_Pr1_Ra10e4/RBC2D_NX64_NZ64_TF50_Pr1_Ra10e4_val.h5"
            self.length = samples
        elif task == "test":
            self.file_data = "/p/project/cexalab/john2/NeuralOperators/RayleighBernardConvection/RBC2D_NX64_NZ64_TF50_Pr1_Ra10e4/RBC2D_NX64_NZ64_TF50_Pr1_Ra10e4_test.h5"
            self.length = samples
        else:
            raise ValueError("task must be in [training,validation,test]")
            
        self.reader = h5py.File(self.file_data, 'r')
        
        # Time
        self.start_t = start_t
        self.end_t = end_t
        self.start = 1
                        
        # Grid size
        self.sx = sx
        self.sz = sz
        
        #Fourier modes 
        self.N_Fourier_F = nf
        
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        # input: vel_start_t, output: vel_end_t
        # inputs = torch.from_numpy(self.reader['snapshots_' + str(index + self.start)+"_t_"+str(self.start_t)]["velocity_t"][:]).type(torch.float32)  # (in_channel, sx, sz)
        # labels = torch.from_numpy(self.reader['snapshots_' + str(index + self.start)+"_t_"+str(self.end_t)]["velocity_t"][:]).type(torch.float32)  # (out_channel, sx, sz)
        
        in_file = self.reader['snapshots_' + str(0 + self.start)+"_t_"+str(index+self.start_t)]
        out_file = self.reader['snapshots_' + str(0 + self.start)+"_t_"+str(index+self.start_t+1)]
        
        # input: [vel_x_start_t, vel_z_start_t, b_start_t] , output: [vel_x_end_t, vel_z_end_t, b_end_t]
        # vel0_x = in_file["velocity_t"][0,:,:]
        # vel0_z = in_file["velocity_t"][1,:,:]
        # b0 = in_file["tasks/buoyancy_t"][:]
        # velt_x = out_file["velocity_t"][0,:,:]
        # velt_z = out_file["velocity_t"][1,:,:]
        # bt = out_file["tasks/buoyancy_t"][:]
        
        # inputs =  torch.from_numpy(np.stack((vel0_x, vel0_z, b0), axis = 0)).type(torch.float32)  # (in_channel=3, sx, sz)
        # labels =  torch.from_numpy(np.stack((velt_x, velt_z, bt), axis = 0)).type(torch.float32)  # (out_channel=3, sx, sz)
 
        # input: [vel_x_start_t, vel_z_start_t, b_start_t, p_start_t] , output: [vel_x_end_t, vel_z_end_t, b_end_t, p_end_t]
        inputs = torch.from_numpy(in_file["input"][:]).type(torch.float32)
        labels = torch.from_numpy(out_file["input"][:]).type(torch.float32)
        
        if self.N_Fourier_F > 0:
            grid = self.get_grid() # [sx, sz, 2]
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            # B shape torch.Size([N_Fourier_F, 2])
            # x_proj torch.Size([sx, sz, N_Fourier_F])
            ff_grid = FF(grid) # [sx, sz, 2*N_Fourier_F]
            ff_grid = ff_grid.permute(2, 0, 1) #[sx, sz, 2*N_Fourier_F] ---> [2*N_Fourier_F,sx,sz]
            inputs = torch.cat((inputs, ff_grid), 0) #([in_channel,sx,sz],[2*N_Fourier,sx,sz]) ---> [2*N_Fourier_F+in_channel,sx,sz]
            

        return inputs.permute(1, 2, 0), labels.permute(1, 2, 0) # [2*N_Fourier_F+in_channel,sx,sz], [out_channel,sx,sz] ---> [sx,sz,2*N_Fourier_F+in_channel], [sx,sz,out_channel]

    def get_grid(self):
        x = torch.linspace(0, 1, self.sx) # [sx]
        z = torch.linspace(0, 1, self.sz) # [sz]

        x_grid, z_grid = torch.meshgrid(x, z) # [sx,sz]

        x_grid = x_grid.unsqueeze(-1)  # [sx,sx,1]
        z_grid = z_grid.unsqueeze(-1)  # [sx,sz,1]
        grid = torch.cat((x_grid, z_grid), -1)  # [sx,sz,2]
        return grid


class RBC2D():
    def __init__(self, 
                 network_properties, 
                 batch_size,
                 device= 'cpu', 
                 training_samples = 250, 
                 start_t = 0, 
                 end_t = 10, 
                 sx= 256, 
                 sz = 64, 
                 in_channels = 2, 
                 out_channels = 2):
        
        self.network_properties =  network_properties 
        self.batch_size = batch_size
        self.device = device
        self.training_samples = training_samples
        self.start_t = start_t
        self.end_t = end_t
        self.sx = sx
        self.sz = sz
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.validation_samples = self.end_t - self.start_t + 1
        self.test_samples = self.end_t - self.start_t + 1
      
        
        self.N_Fourier_F = self.network_properties["FourierF"]
        
        self.model = FNO2d(fno_architecture = self.network_properties, 
                            in_channels = self.in_channels + 2 * self.N_Fourier_F, 
                            out_channels = self.out_channels, 
                            device= self.device) 
        

        num_workers = 1
        
        self.train_loader = DataLoader(RBCDataset2D("training", self.N_Fourier_F, self.training_samples, self.start_t, self.end_t, self.sx, self.sz), batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(RBCDataset2D("validation", self.N_Fourier_F, self.validation_samples, self.start_t, self.end_t, self.sx, self.sz), batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(RBCDataset2D("test", self.N_Fourier_F, self.test_samples, self.start_t, self.end_t, self.sx, self.sz), batch_size=self.batch_size, shuffle=False, num_workers=num_workers)