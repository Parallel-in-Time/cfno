import random

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from fourier_operator import FourierFeatures,FNO2d

from torch.utils.data import Dataset

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# NOTE:
#All the training sets should be in the folder: data/


def activation_selection(choice):
    if choice in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif choice in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif choice in ['lrelu', 'LReLU']:
        return nn.LeakyReLU(inplace=True)
    elif choice in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif choice in ['softplus', 'Softplus']:
        return nn.Softplus(beta=4)
    elif choice in ['celu', 'CeLU']:
        return nn.CELU()
    elif choice in ['elu']:
        return nn.ELU()
    elif choice in ['mish']:
        return nn.Mish()
    else:
        raise ValueError('Unknown activation function')
        
        
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
        training_properties["training_samples"] = 256
        
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
        
#         self.model = FNO1d(fno_architecture = network_properties,
#                            nfun=1,
#                            padding_frac=1/4,
#                            device=device)

        num_workers = 8
        
        self.train_loader = DataLoader(WaveEquationDataset("training", self.N_Fourier_F, training_samples, 5, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(WaveEquationDataset("validation", self.N_Fourier_F, training_samples, 5, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(WaveEquationDataset("test", self.N_Fourier_F, training_samples, 5, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        
        
        
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
                           # nfun=1,
                           # padding_frac=1/4,
                           # device=device)
                           

        num_workers = 8

        self.train_loader = DataLoader(DarcyDataset("training", self.N_Fourier_F, training_samples), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(DarcyDataset("validation", self.N_Fourier_F, training_samples), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(DarcyDataset("testing", self.N_Fourier_F, training_samples, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)