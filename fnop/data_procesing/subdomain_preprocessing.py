import os
import sys
sys.path.insert(1, os.getcwd())
import h5py
from fnop.data_procesing.data_loader import FNOSubDomain

dim = 'FNO2D'
start_time = 50.0
stop_time = 100.0
dt = 0.1
gridx = 256
gridy = 64
ndomain_x = 4
ndomain_y = 1
tStep = 1
T_in = 1
T = 1

domain = FNOSubDomain(start_time=start_time,
                      stop_time=stop_time,
                      gridx=gridx,
                      gridy=gridy,
                      ndomain_x=ndomain_x,
                      ndomain_y=ndomain_y,
                      dt=dt,
                      dim=dim,
                      tStep=tStep,
                      T_in=T_in,
                      T=T
                    )

train_state, train_time = domain.input_loader(root_path='/p/scratch/cexalab/john2/RBC2D_data_dt1e_1/RBC2D_NX256_NZ64_TI0_TF150_Pr1_Ra1e7_dt0_1_train',
                                              samples=10)
print(f'Train state shape: {train_state.shape}')
print(f'Train time: {train_time.shape}')

train_domain_state = domain.state_split_subdomain(input_state=train_state)
print(f'Domain train split state: {train_domain_state.shape}')


result_path = '/p/project1/cexalab/john2/NeuralOperators/RayleighBernardConvection/processed_data/RBC2D_NX4_64_NZ1_64_TI50_TF100_Pr1_Ra1e7_dt0_1.h5'
reader = h5py.File(result_path, mode='a')
reader['train'] = train_domain_state
reader['train_time'] = train_time

val_state, val_time = domain.input_loader(root_path='/p/scratch/cexalab/john2/RBC2D_data_dt1e_1/RBC2D_NX256_NZ64_TI0_TF150_Pr1_Ra1e7_dt0_1_val',
                                          samples=5)
print(f'Val state shape: {val_state.shape}')
print(f'Val time: {val_time.shape}')

val_domain_state = domain.state_split_subdomain(input_state=val_state)
print(f'Domain split state: {val_domain_state.shape}')

reader['val'] = val_domain_state
reader['val_time'] = val_time

# (jureca_no) [john2@jrlogin07.jureca]$python fnop/data_procesing/subdomain_preprocessing.py 
# Train state shape: (1501, 10, 4, 256, 64)
# Train time: (500, 10)
# Domain train split state: (500, 40, 256, 64)
# Val state shape: (1501, 5, 4, 256, 64)
# Val time: (500, 5)
# Domain split state: (500, 20, 256, 64)