import h5py
import torch
from timeit import default_timer
from fnop.utils import UnitGaussianNormalizer


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
        Data loader for FNO2D recurrent in time model

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
    
    