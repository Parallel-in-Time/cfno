import h5py
import torch
import glob
import numpy as np
from timeit import default_timer
from fnop.utils import UnitGaussianNormalizer
from fnop.data_procesing.data_utils import check_subdomain


class FNODataLoader():
    """
    Train and validation data loader 
    """
    def __init__(self,
                 batch_size:int,
                 gridx:int,
                 gridy:int,
                 dt:float,
                 dim:str,
                 xStep:int=1,
                 yStep:int=1,
                 tStep:int=1,
                 start_index:int=0,
                 T_in:int=1,
                 T:int=1,
                 **kwargs
    ):
        """

        Args:
            batch_size (int): batch size 
            gridx (int): size of gridx
            gridy (int): size of gridy
            dt (float): delta timestep 
            dim (str): FNO2D or FNO3D strategy
            xStep (int): slicing for x-grid. Defaults to 1.
            yStep (int): slicing for y-grid. Defaults to 1.
            tStep (int): time slice. Defaults to 1.
            start_index (int): time start index. Defaults to 0.
            T_in (int):number of input timesteps. Defaults to 1.
            T (int): number of output timesteps. Defaults to 1.
        """
        super().__init__()
        self.batch_size = batch_size
        self.start_index = start_index
        self.dim = dim
        self.gridx = gridx
        self.gridy = gridy
        self.dt = dt
        self.xStep = xStep
        self.yStep = yStep
        self.tStep = tStep
        self.T_in = T_in
        self.T = T
        
        self.gridx_state = 4*self.gridx  # stacking [velx,velz,buoyancy,pressure]
  
           
    def data_loader(self, task:str, nsamples:int, reader):
        """
        Data loader for FNO model

        Args:
            task (str): 'train', or 'val' or 'test
            nsamples (int): number of simulation samples
            reader: hdf5 file reader

        Returns:
            data_loader (torch.utils.data.DataLoader()): data loader 
            
        """
        if task == 'train':
            shuffle = True
        else:
            shuffle = False
            
        inputs = torch.tensor(reader[task][:nsamples, ::self.xStep, ::self.yStep, \
                                           self.start_index: self.start_index + (self.T_in*self.tStep): self.tStep], \
                                           dtype=torch.float)
        
        outputs = torch.tensor(reader[task][:nsamples, ::self.xStep, ::self.yStep, \
                                            self.start_index + (self.T_in*self.tStep): self.start_index + \
                                            (self.T_in + self.T)*self.tStep: self.tStep],\
                                            dtype=torch.float)

        print(f"input data for {task}:{inputs.shape}")
        print(f"output data for {task}: {outputs.shape}")
        assert (self.gridx_state == outputs.shape[-3])
        assert (self.gridy == outputs.shape[-2])
        assert (self.T ==outputs.shape[-1])
        
        if self.dim == 'FNO3D':
            # input_normalizer = UnitGaussianNormalizer(inputs)
            # inputs = input_normalizer.encode(inputs)
            # output_normalizer = UnitGaussianNormalizer(outputs)
            # outputs = output_normalizer.encode(outputs)
            
            inputs = inputs.reshape(nsamples, self.gridx_state, self.gridy, 1, self.T_in).repeat([1,1,1,self.T,1])
            print(f"Input data after reshaping for {task}:{inputs.shape}")
        

        data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(inputs, outputs), batch_size=self.batch_size, shuffle=shuffle)
        
        return data_loader


class FNOSubDomain():
    """
    Processing Dedalus data into sub-domain
    """
    def __init__(self,
                 start_time:float,
                 stop_time:float,
                 gridx:int,
                 gridy:int,
                 ndomain_x:int,
                 ndomain_y:int,
                 dt:float,
                 dim:str,
                 tStep:int=1,
                 T_in:int=1,
                 T:int=1,
                 **kwargs
    ):
        """

        Args:
            start_time (float): start simulation time
            stop_time (float): stop simulation time
            gridx (int): size of gridx
            gridy (int): size of gridy
            ndomain_x (int): number of sub-domain on x-grid
            ndomain_y (int): number of sub-domain on y-grid
            dt (float): delta timestep 
            dim (str): FNO2D or FNO3D strategy
            tStep (int): time slice. Defaults to 1.
            T_in (int):number of input timesteps. Defaults to 1.
            T (int): number of output timesteps. Defaults to 1.
        """
        super().__init__()
        self.start_time = start_time
        self.stop_time = stop_time
        self.dim = dim
        self.gridx = gridx
        self.gridy = gridy
        self.ndomain_x = ndomain_x
        self.ndomain_y = ndomain_y
        self.dt = dt
        self.tStep = tStep
        self.T_in = T_in
        self.T = T
        self.subdomain_x = check_subdomain(self.gridx, self.ndomain_x)
        self.subdomain_y = check_subdomain(self.gridy, self.ndomain_y)
        self.start_time_index = int(self.start_time/self.dt)
        self.stop_time_index = int(self.stop_time/self.dt)
    
    def input_loader(self,
                     root_path:str,
                     samples:int,
    ):
        """
        Load data from dedalus file into FNO input format

        Args:
            root_path (str): root dir path of dedalus data
            samples (int): number of data files
        Returns:
           input_state (np.ndarray): input state with shape [time, samples, 4, gridx, gridy]
           input_time (np.ndarray): input time with shape [time, samples]
        """
        state = []
        time  = []

        for i in range(0, samples):
            file_dir = f'{root_path}_{i+1}'
            files = glob.glob(f'{file_dir}/*.h5')
            sim_time = []
            for f in files:
                data =  h5py.File(f, mode='r')
                sim_time.append(data['scales/sim_time'][self.start_time_index:self.stop_time_index ])
                vel_x = data['tasks/velocity'][:,0,:,:]
                vel_y = data['tasks/velocity'][:,1,:,:]
                # print(vel_x.shape, vel_y.shape)
                buoyancy = data['tasks/buoyancy']
                pressure = data['tasks/pressure']
                state.append(np.stack([vel_x, vel_y, buoyancy, pressure], axis=0))
            time.append(sim_time)
        input_state = np.array(state)
        state_dim = input_state.shape
        input_time = np.vstack(time).transpose()
        input_state = input_state.reshape(state_dim[2], state_dim[0], state_dim[1], state_dim[3], state_dim[4])
        # [time, samples, 4, gridx, gridy]
        # [time, samples]
        return input_state, input_time
    
    def state_split_subdomain(self,input_state: np.ndarray):
        """
        Split the input state into sub domains

        Args:
            input_state (np.ndarray): state with shape [time, samples, 4, gridx, gridy]

        Returns:
            state (np.ndarray): state with shape  [time, ndomain_y*ndomain_x*samples, 4*subdomain_x, subdomain_y]
        """
    
        task_x = []
        task_y = []
        for t in range(self.start_time_index, self.stop_time_index, self.tStep):
            # [time, samples, 4, gridx, gridy]
            sub_taskx = []
            for i in range(self.ndomain_x):
                taskx = input_state[t, :,:, i*self.subdomain_x: i*self.subdomain_x + self.subdomain_x, :]
                # [samples, 4, sub_domainx, gridy]
                sub_taskx.append(taskx) 
                # [ndomain_x, samples, 4, subdomain_x, gridy]
            task_x.append(sub_taskx)
            
        dim_x = np.array(task_x).shape  
        # [time, ndomain_x, samples, 4, subdomain_x, gridy]
        state_x = np.array(task_x).reshape(dim_x[0], dim_x[2], dim_x[1], dim_x[3], dim_x[4], dim_x[5]) 
        # [time, samples, ndomain_x, 4, subdomain_x, gridy]
        for t in range(state_x.shape[0]):
            sub_tasky = []
            for j in range(self.ndomain_y):
                tasky = state_x[t, :, :, :, :, j*self.subdomain_y: j*self.subdomain_y + self.subdomain_y]
                # [samples, ndomain_x, 4, subdomain_x, sub_domain_y]
                sub_tasky.append(tasky)
                # [ndomainy, samples, ndomain_x, 4, subdomain_x, subdomain_y]
            task_y.append(sub_tasky)

        dim_y = np.array(task_y).shape  
        # [time, ndomain_y, samples, ndomain_x, 4, subdomain_x, subdomain_y]
        state = np.array(task_y).reshape(dim_y[0], dim_y[2]*dim_y[1]*dim_y[3], dim_y[4]*dim_y[5], dim_y[6]) 
        # [time, ndomain_y*ndomain_x*samples, 4*subdomain_x, subdomain_y]
        return state
    
    def subdomain_data_loader(self, task:str, nsamples:int, batch_size:int, reader):
        """
        Sub-domain data loader for FNO model

        Args:
            task (str): 'train', or 'val' or 'test
            nsamples (int): number of simulation samples
            batch_size (int): batch size 
            reader: hdf5 file reader

        Returns:
            data_loader (torch.utils.data.DataLoader()): data loader 
            
        """
        if task == 'train':
            shuffle = True
        else:
            shuffle = False
        a = []
        u = []
        timestep = self.tStep
        ntimes = reader[task].shape[0]
        for index in range(0, ntimes-self.T, timestep):
            a.append(torch.tensor(reader[task][index: index + (self.T_in*self.tStep): self.tStep,
                                         :nsamples, :,:], \
                                         dtype=torch.float))
            
            u.append(torch.tensor(reader[task][index + (self.T_in*self.tStep): index + \
                                         (self.T_in + self.T)*self.tStep: self.tStep,\
                                         :nsamples, :,:], \
                                         dtype=torch.float))
        a = torch.as_tensor(np.array(a))
        u = torch.as_tensor(np.array(u))
        print(f"input data for {task} (before reshape):{a.shape}")
        print(f"output data for {task} (before_reshape): {u.shape}")
        inputs = a.reshape(a.shape[0]*a.shape[2], a.shape[3], a.shape[4], a.shape[1])
        outputs = u.reshape(u.shape[0]*u.shape[2], u.shape[3], u.shape[4], u.shape[1])
        print(f"input data for {task}:{inputs.shape}")
        print(f"output data for {task}: {outputs.shape}")
        assert (4*self.subdomain_x == outputs.shape[-3])
        assert (self.subdomain_y == outputs.shape[-2])
        assert (self.T == outputs.shape[-1])
        
        if self.dim == 'FNO3D':
            # input_normalizer = UnitGaussianNormalizer(inputs)
            # inputs = input_normalizer.encode(inputs)
            # output_normalizer = UnitGaussianNormalizer(outputs)
            # outputs = output_normalizer.encode(outputs)
            
            inputs = inputs.reshape(inputs.shape[0], 4*self.subdomain_x, self.subdomain_y, 1, self.T_in).repeat([1,1,1,self.T,1])
            print(f"Input data after reshaping for {task}:{inputs.shape}")
        

        data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(inputs, outputs), batch_size=batch_size, shuffle=shuffle)
        
        return data_loader