"""
Process FNO model data
    
Usage:
    python data_processing.py
        --config=<config_file>
                 
"""


import os
import sys
sys.path.insert(1, os.getcwd())
import h5py
from fnop.data_procesing.data_loader import FNOSubDomainData, FNOData
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
import argparse
from timeit import default_timer

parser = argparse.ArgumentParser(description='FNO Data processing')
parser.add_argument('--config_file', type=str,
                    help='FNO config yaml file')

args = parser.parse_args()
pipe = ConfigPipeline(
        [
          YamlConfig(args.config_file),
          ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        ]
    )
config = pipe.read_conf()
model_config = config.FNO
data_config = config.data
domain_config = data_config.subdomain_args

domain = FNOSubDomainData(start_time=domain_config.start_time,
                            stop_time=domain_config.stop_time,
                            gridx=data_config.gridx,
                            gridy=data_config.gridy,
                            ndomain_x=domain_config.ndomain_x,
                            ndomain_y=domain_config.ndomain_y,
                            dt=data_config.dt,dim=config.dim,
                            tStep=data_config.tStep,T_in=model_config.T_in,
                            T=model_config.T
                          )

loader = FNOData(gridx= data_config.gridx, gridy=data_config.gridy,
                  dt=data_config.dt, dim=config.dim, xStep=data_config.xStep, 
                  yStep=data_config.yStep, tStep=data_config.tStep,
                  start_index=data_config.start_index, 
                  stop_index=data_config.stop_index, 
                  timestep=data_config.timestep,
                  T_in=model_config.T_in, T=model_config.T
                  )
if data_config.subdomain_process:
  print(f'Splitting data into {domain_config.ndomain_x} subdomains along x-grid and {domain_config.ndomain_y} sub-domains along y-grid')
  start_domain = default_timer()
  train_state, train_time = domain.input_loader(root_path=domain_config.train_root_path,
                                                samples=data_config.train_samples)
  print(f'Train state shape: {train_state.shape}') # [time, samples, 4, gridx, gridy]
  print(f'Train time: {train_time.shape}')   
       

  train_domain_state = domain.state_split_subdomain(input_state=train_state)
  print(f'Domain train split state: {train_domain_state.shape}')
  

  val_state, val_time = domain.input_loader(root_path=domain_config.val_root_path,
                                            samples=data_config.val_samples)
  print(f'Val state shape: {val_state.shape}')
  print(f'Val time: {val_time.shape}')

  val_domain_state = domain.state_split_subdomain(input_state=val_state)
  print(f'Domain val split state: {val_domain_state.shape}')
  reader = h5py.File(domain_config.subdomain_data_path, mode='a')
  reader['train'] = train_domain_state    # [time, ndomain_y*ndomain_x*samples, 4*subdomain_x, subdomain_y]
  reader['train_time'] = train_time       # [time, samples]
  reader['val'] = val_domain_state
  reader['val_time'] = val_time
  reader.close()
  
  print(f'Time taken for domain splitting (s): {default_timer()-start_domain}')
  

start_processing = default_timer()
if data_config.subdomain_process:
  reader = h5py.File(domain_config.subdomain_data_path, mode='r+')
  train_subdomain_samples = reader['train'].shape[1]
  val_subdomain_samples = reader['val'].shape[1]
  train_input, train_output = domain.subdomain_data_loader('train', train_subdomain_samples, reader, data_config.multistep)
  val_input, val_output = domain.subdomain_data_loader('val', val_subdomain_samples, reader, data_config.multistep)
  reader.close()
else:
  train_reader = h5py.File(data_config.train_data_path, mode="r")
  val_reader = h5py.File(data_config.val_data_path, mode="r") 
  train_input, train_output = loader.data_loader('train', data_config.train_samples, train_reader, data_config.multistep)
  val_input, val_output = loader.data_loader('val', data_config.val_samples, val_reader, data_config.multistep)
  train_reader.close()
  val_reader.close()
  
# print(f'train input shape: {train_input.shape}, {type(train_input)}')
# print(f'train output shape: {train_output.shape}, {type(train_output)}')
# print(f'val input shape: {val_input.shape}, {type(val_input)}')
# print(f'val output shape: {val_output.shape}, {type(val_output)}')

save_multidata = h5py.File(data_config.processed_data_path, mode="a")
save_multidata['train_inputs'] = train_input
save_multidata['train_outputs'] = train_output
save_multidata['val_inputs'] = val_input
save_multidata['val_outputs'] = val_output
save_multidata.close()

print(f'Time taken for pre-processing data (s): {default_timer()-start_processing}')
  

# inputs:[(end_index-start_index-(T_in+T)*tStep)/timestep *ndomain_y*ndomain_x*samples,
#          4*subdomain_x, subdomain_y, T_in]
# outputs:[(end_index-start_index)-(T_in+T)*tStep)/timestep *ndomain_y*ndomain_x*samples,
#          4*subdomain_x, subdomain_y, T]