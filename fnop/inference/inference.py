""""
Perform inference and plot results for 2D Rayleigh Benard Convection

Usage:
    python inference.py 
        --run=<run_tracker> 
        --model_checkpoint=<checkpoint> 
        --train_data_path=<train_data_path> (only when using FNO3D) 
        --test_data_path=<test_data_path>  
        --dim=FNO2D or FNO3D 
        --modes=12 
        --width=20 
        --batch_size=50 
        --rayleigh=1.5e7 
        --prandtl=1.0
        --gridx=256 
        --gridy=64 
        --train_samples=<train_samples> (only when using FNO3D) 
        --nTest=<test_samples> 
        --T_in=<input timesteps 
        --T=<output_timesteps> 
        --start_index=<dedalus_start_index> 
        --stop_index=<dedalus_stop_index> 
        --timestep=<dedalus_time_slice> 
        --dedalus_time_index=<absolute_dedalus_time_index>
        --dt=<dedalus_data_dt>
        --folder=<results>  
        --calc_loss
        
    optional args:
        --single_data_path<hdf5 file contains train, val and test data>
        --plotFile (only to plot without model_inference)
        --store_result 
        
"""

import os
import h5py
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from timeit import default_timer
import torch
from fno3d import FNO3d
from fno2d_recurrent import FNO2d
from NeuralOperators.neural_operators.fnop.utils import LpLoss
from inference_plot import cross_section_plots, cross_section_error_plot
from preprocessing import data_loading, time_extract
from postprocessing import state_extract, save_inference

def inference_with_loss(test_loader,loss,model,
                        gridx:int, gridy:int,
                        dim:str, batch_size:int=1,
                        tStep:int=1,T:int=1,
                        device:str='cpu',y_normalizer=None):
    """
    Perform inference with loss calulations

    Args:
        test_loader (torch.DataLoader): test dataloader 
        loss (func): loss function
        model (torch.nn.Module) : FNO model
        gridx (int): x grid size
        gridy (int): y grid size
        dim (str): FNO2D or FNO3D strategy
        batch_size (int): inference batch size
        tStep (int, optional): timestep slicing. Defaults to 1.
        T (int, optional): number of output timesteps. Defaults to 1.
        device (str, optional): Defaults to 'cpu'.
        y_normalizer (func, optional): test data normalizer. Defaults to None.

    Returns:
        inputs (list): Dedalus input stack [velx, velz, buoyancy, pressure]
        outputs (list): Dedalus output stack [velx, velz, buoyancy, pressure]
        predictions (list): FNO output stack [velx, velz, buoyancy, pressure]
    """
   
    if dim == 'FNO3D':
        fno3d_loss = 0
    else:
        fno2d_full_loss = 0
        fno2d_step_loss = 0
        
    pred = torch.zeros([batch_size, gridx, gridy, T])
    inputs = []
    outputs = []
    predictions = []
        
    for step, (xx, yy) in enumerate(tqdm(test_loader)):
        xx, yy = xx.to(device), yy.to(device)
        xx_org = xx
        if dim == 'FNO3D':
            out = model(xx_org).view(batch_size, gridx, gridy, T)
            out = y_normalizer.decode(out)
            fno3d_loss += loss(out.view(1, -1), yy.view(1, -1)).item()
            pred = out 
        else:
            step_loss = 0
            for t in range(0, T, tStep):
                y = yy[..., t:t + tStep]   
                im = model(xx)
                step_loss += loss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
                
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)
        
                xx = torch.cat((xx[..., tStep:], im), dim=-1)
            
            fno2d_step_loss += step_loss.item()
            fno2d_full_loss += loss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
    
        inputs.append(xx_org)
        outputs.append(yy)
        predictions.append(pred)
        
        if step == 0 or step %10 == 0:
            print(f"Batch:{step} , xx:{xx_org.shape}, yy:{yy.shape}, pred:{pred.shape}")
            if dim == 'FNO3D':
                print(f"FNO3D loss: {fno3d_loss/((step+1)*batch_size)}")
            else:
                print(f"FNO2D step loss: {fno2d_step_loss/((step+1)*batch_size)/ (T/tStep)}")
                print(f"FNO2D full loss: {fno2d_full_loss/((step+1)*batch_size)}")
    
    return inputs, outputs, predictions

def inference_without_loss(model, input_data,
                          gridx:int, gridy:int,
                          dim:str,batch_size:int=1,
                          tStep:int=1,T:int=1,
                          device:str='cpu'):
    """
    Perform inference with loss calulations

    Args:
        model (torch.nn.Module) : FNO model
        input_data (torch.tensor): FNO input tensor stack [velx, velz, buoyancy, pressure]
        gridx (int): x grid size
        gridy (int): y grid size
        dim (str): FNO2D or FNO3D strategy
        batch_size (int): inference batch size
        tStep (int, optional): timestep slicing. Defaults to 1.
        T (int, optional): number of output timesteps. Defaults to 1.
        device (str, optional): Defaults to 'cpu'.

    Returns:
        xx_org (torch.tensor): Dedalus input stack [velx, velz, buoyancy, pressure]
        pred (torch.tensor): FNO output stack [velx, velz, buoyancy, pressure]
    """
   
    
    pred = torch.zeros([batch_size, gridx, gridy, T])
    xx = input_data.to(device)
    xx_org = xx
    if dim == 'FNO3D':
        out = model(xx_org).view(batch_size, gridx, gridy, T)
        # out = y_normalizer.decode(out)
        pred = out 
    else:
        for t in range(0, T, tStep):
            im = model(xx)
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
    
            xx = torch.cat((xx[..., tStep:], im), dim=-1)
    

    return xx_org, pred
    
def model_inference(checkpoint:str, test_data_path:str, dim:str, modes:int, width:int,
                    gridx:int, gridy:int, batch_size:int, nTest:int=1,
                    xStep:int=1,yStep:int=1,tStep:int=1, index:int=0,
                    T_in:int=1, T:int=1, calc_loss:bool=False, loss=None,
                    device:str='cpu',single_data_path:bool=False, **kwargs):
    """
    Function to perform FNO model inference

    Args:
        checkpoint (str): path to FNO model checkpoint
        test_data_path (str): path to hdf5 test data file.
        dim (str): FNO2D or FNO3D strategy
        modes (int): number of fourier modes in the model
        width (int): number of neurons in the FNO model
        loss (func): loss function. Defaults to None
        calc_loss (bool): perform inference with loss
        gridx (int): x grid size
        gridy (int): y grid size
        batch_size (int): inference batch size
        xStep (int, optional): x-grid slicing. Defaults to 1.
        yStep (int, optional): y-grid slicing. Defaults to 1.
        tStep (int, optional): timestep slicing. Defaults to 1.
        index (int, optional): start index for T_in
        T_in (int, optional): number of input timesteps. Defaults to 1.
        T (int, optional): number of output timesteps. Defaults to 1.
        device (str, optional): Defaults to 'cpu'.
        nTest (int, optional): number of testing samples. Defaults to 1.
        single_data_path (bool): hdf5 file contains test and/or train data
        
        Optional:
        train_data_path (str): path to hdf5 train data file
        train_samples (int): number of training samples 

    Returns:
        inputs (np.ndarray): dedalus inputs
        predictions (np.ndarray): FNO outputs
        outputs: Dedalus outputs
    """
    
    y_normalizer = None
    inference_func_start = default_timer()
    print(f'Entered model_inference() at {inference_func_start}')

    # Model
    if dim == 'FNO3D':
        model = FNO3d(modes, modes, modes, width, T_in, T).to(device)
    else:
        model = FNO2d(modes, modes, width, T_in, T).to(device)

    model_checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    if 'model_state_dict' in model_checkpoint.keys():
        model.load_state_dict(model_checkpoint['model_state_dict'])
    else:
        model.load_state_dict(model_checkpoint)
    
    if 'train_samples' in kwargs.keys():
        train_samples = kwargs['train_samples']
    if 'train_data_path' in kwargs.keys():
        train_data_path = kwargs['train_data_path']
        
    # Inference
    print('Starting model inference...')
    inference_time_start = default_timer()

    model.eval()
    with torch.no_grad():
        if calc_loss:
            if single_data_path:
                if dim == 'FNO3D':
                    test_loader, y_normalizer = data_loading(dim, test_data_path, batch_size, gridx, gridy, xStep, yStep, tStep, T_in, T, nTest, index, single_data_path, train_samples=train_samples)
                else:
                    test_loader = data_loading(dim, test_data_path, batch_size, gridx, gridy, xStep, yStep, tStep, T_in, T, nTest, index, single_data_path)
            else:
                if dim == 'FNO3D':
                    test_loader, y_normalizer = data_loading(dim, test_data_path, batch_size, gridx, gridy, xStep, yStep, tStep,T_in, T, nTest, index, single_data_path, train_data_path=train_data_path, train_samples=train_samples)
                else:
                    test_loader = data_loading(dim, test_data_path, batch_size, gridx, gridy, xStep, yStep, tStep,T_in, T, nTest, index, single_data_path)
    
            inputs, outputs, predictions = inference_with_loss(test_loader, loss, model, 
                                                               gridx, gridy, dim,
                                                               batch_size, tStep,  T, 
                                                               device,y_normalizer)
        else:
            test_reader = h5py.File(test_data_path, mode="r")
            dataloader_time_start = default_timer()
            input_data = torch.tensor(test_reader['test'][0, ::xStep, ::yStep, index: index + (T_in*tStep): tStep], dtype=torch.float)
            dataloader_time_stop = default_timer()
            print(f'Total time taken for dataloading (s): {dataloader_time_stop - dataloader_time_start}')
            inputs, predictions = inference_without_loss(model, input_data, 
                                                         gridx, gridy, dim, 
                                                         batch_size,
                                                         tStep,T, 
                                                         device)
          
    predictions_cpu = torch.stack(predictions).cpu()
    inference_time_stop = default_timer()
    print(f'Total time taken for model inference for {T} steps of {nTest} samples with batchsize {batch_size} on {device} (s): {inference_time_stop - inference_time_start}')
    
    inputs_cpu = torch.stack(inputs).cpu()
    inference_func_stop = default_timer()
    print(f'Exiting model_inference()...')
    print(f'Total time in model_inference() function (s): {inference_func_stop - inference_func_start}')
    
    if calc_loss:
        outputs_cpu = torch.stack(outputs).cpu()
        return np.array(inputs_cpu), np.array(outputs_cpu), np.array(predictions_cpu)
    else:
        return np.array(inputs_cpu), np.array(predictions_cpu)

def main( folder:str, dim:str, model_checkpoint:str, 
          modes:int, width:int, batch_size:int, 
          rayleigh:float, prandtl:float,
          gridx:int, gridy:int, 
          start_index:int, stop_index:int,
          timestep:int, dedalus_time_index:int,dt:int,
          test_data_path:str, train_data_path:str, single_data_path:str,
          nTest:int=1, train_samples=None,
          xStep:int=1, yStep:int=1,tStep:int=1,
          T_in:int=1, T:int=1,run:int=1,
          calc_loss:bool=False, plotFile:bool=False, store_result:bool=True,
          *args,**kwargs):
    """

    Args:
        folder (str): root path for storing inference results
        dim (str): FNO2D or FNO3D strategy
        model_checkpoint (str): path to model checkpoint
        modes (int): number of fourier modes
        width (int): number of neurons in layers
        batch_size (int): inference batch size 
        rayleigh (float):Rayleigh number 
        prandtl (float): Prandtl number 
        gridx (int): x grid size
        gridy (int): y grid size
        start_index (int): start index for input time
        stop_index (int): stop index for input time
        timestep (int): interval for input time
        dedalus_time_index (int): absolutime time index 
        dt (int): dedalus timestep
        calc_loss (bool): to calculate inference loss
        plotFile (bool): to plot crossectional plots from a Hdf5 file
        store_result (bool): to store inference result to a Hdf5 file 
        nTest (int, optional): number of inference samples. Defaults to 1.
        train_samples (int, optional): number of training samples. Defaults to None.
        xStep (int, optional): x-grid slicing. Defaults to 1.
        yStep (int, optional): y-grid slicing. Defaults to 1.
        tStep (int, optional): timestep slicing. Defaults to 1.
        T_in (int, optional): number of input timesteps. Defaults to 1.
        T (int, optional): number of output timesteps. Defaults to 1.
        run (int, optional): run tracker. Defaults to 1.
        single_data_path (str): path to hdf5 file containing test and/or train data
        test_data_path (str): path to hdf5 test data file.
        train_data_path (str): path to hdf5 train data file
    """
    
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    # since velx,vely,pressure and buoyacy are stacked
    gridx_state = 4*gridx
 
    fno_path = Path(f'{folder}/rbc_{dim}_N{nTest}_m{modes}_w{width}_bs{batch_size}_dt{dt}_tin{T_in}_inference_{device}_run{run}')
    fno_path.mkdir(parents=True, exist_ok=True)

    if plotFile:
        cross_section_plots(plotFile, gridx, gridy, dim, fno_path, rayleigh, prandtl, T_in, T)  
    else:
        for iteration, index in enumerate(range(start_index, stop_index, timestep)):
            start_index_org =  dedalus_time_index + index
            time_in, time_out = time_extract(start_index_org, dt, T_in, T, tStep)
            
            if calc_loss:
                if single_data_path:
                    inputs, outputs, predictions = model_inference(model_checkpoint, test_data_path, dim, modes, width,
                                                                    gridx_state, gridy, batch_size, nTest,
                                                                    xStep, yStep, tStep, index,
                                                                    T_in, T, calc_loss, LpLoss(size_average=False),
                                                                    device, single_data_path, train_samples=train_samples)
                else:
                    if dim =='FNO3D':
                        inputs, outputs, predictions = model_inference(model_checkpoint, test_data_path, dim, modes, width,
                                                                        gridx_state, gridy, batch_size, nTest,
                                                                        xStep, yStep, tStep, index,
                                                                        T_in, T, calc_loss, LpLoss(size_average=False),
                                                                        device, single_data_path, train_data_path=train_data_path, train_samples=train_samples)
                    else: 
                        inputs, outputs, predictions = model_inference(model_checkpoint, test_data_path, dim, modes, width,
                                                                        gridx_state, gridy, batch_size, nTest,
                                                                        xStep, yStep, tStep, index,
                                                                        T_in, T, calc_loss, LpLoss(size_average=False),
                                                                        device, single_data_path)
                    
                print(f"Model Inference: Input{inputs.shape}, Output{outputs.shape}, Prediction{predictions.shape}")
            else:
                inputs, predictions = model_inference(model_checkpoint, test_data_path, dim, modes, width,
                                                      gridx_state, gridy, batch_size, nTest,
                                                      xStep, yStep, tStep, index,
                                                      T_in, T, calc_loss, LpLoss(size_average=False),
                                                      device, single_data_path)
                print(f"Model Inference: Input{inputs.shape}, Prediction{predictions.shape}")
            
            # taking results for a random sample when batchsize > 1
            batches = predictions.shape[0]
            batchsize = predictions.shape[1]
            batch_num = np.random.randint(0,batches)
            sample = np.random.randint(0,batchsize)
            
            if dim == "FNO3D":
                # since test_a = test_a.reshape(nTest, gridx, gridy, 1, T_in).repeat([1,1,1,T,1])
                ux, uy, b_in, p_in = state_extract(inputs[batch_num, sample, :, :, 0, :], gridx, gridy, T_in)
            else:
                ux, uy, b_in, p_in = state_extract(inputs[batch_num, sample, :, :, :], gridx, gridy, T_in)
                
            vx_pred, vy_pred, b_pred, p_pred = state_extract(predictions[batch_num, sample, :, :, :], gridx, gridy,T)
           
            print(f"Inference for Batch Number: {batch_num}, Sample: {sample}") 
            if store_result:
                if calc_loss:
                    vx, vy, b_out, p_out = state_extract(outputs[batch_num, sample, :, :, :], gridx, gridy, T)
                    save_inference(iteration, ux, vx_pred, 
                                    uy, vy_pred,
                                    b_in, b_pred, p_in, p_pred, 
                                    time_in, time_out,f'{fno_path}/inference.h5', calc_loss,
                                    xStep, yStep, tStep,
                                    vx, vy, b_out, p_out)
                    
                    cross_section_error_plot(vx, vx_pred,
                                            vy, vy_pred,
                                            b_out, b_pred,
                                            p_out, p_pred,
                                            time_in, time_out, 
                                            dim, fno_path, 
                                            gridx, gridy, 
                                            rayleigh, prandtl)
                else:
                    save_inference(iteration, ux, vx_pred, uy, vy_pred,
                                    b_in, b_pred, p_in, p_pred, 
                                    time_in, time_out,f'{fno_path}/inference.h5', calc_loss,
                                    xStep, yStep, tStep)
            else:
                print(f'x-vel: {vx_pred} \n y-vel: {vy_pred} \n buoyancy: {b_pred} \n pressure: {p_pred}')
                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FNO Inference')
    parser.add_argument('--run', type=int, default=1,
                            help='training tracking number')
    parser.add_argument('--model_checkpoint', type=str,
                        help=" Torch model state path")
    parser.add_argument('--single_data_path', action='store_true',
                        help='if hdf5 file contain train, val and test data')
    parser.add_argument('--train_data_path', type=str,default=None,
                        help='path to train data hdf5 file')
    parser.add_argument('--test_data_path', type=str,default=None,
                        help='path to test data hdf5 file')
    parser.add_argument('--dim', type=str,default="FNO2D",
                        help="FNO2D+recurrent time or FNO3D")
    parser.add_argument('--modes', type=int, default=12,
                        help="Fourier modes")
    parser.add_argument('--width', type=int, default=20,
                        help="Width of layer")
    parser.add_argument('--batch_size', type=int, default=5,
                        help="Batch size")
    parser.add_argument('--rayleigh', type=float, default=1.5e7,
                        help="Rayleigh Number")
    parser.add_argument('--prandtl', type=float, default=1.0,
                        help="Prandtl Number")
    parser.add_argument('--gridx', type=int, default=256,
                        help="size of x-grid")
    parser.add_argument('--gridy', type=int, default=64,
                        help="size of y-grid")
    parser.add_argument('--T_in', type=int, default=1,
                        help='number of input timesteps to FNO')
    parser.add_argument('--T', type=int, default=1,
                        help='number of output timesteps to FNO')
    parser.add_argument('--dedalus_time_index', type=int, 
                        help='absolute time index for dedalus data')
    parser.add_argument('--start_index', type=int, 
                        help='relative time index for dedalus data')
    parser.add_argument('--stop_index', type=int, 
                        help='relative time index for dedalus data')
    parser.add_argument('--timestep', type=int, 
                        help='slicer for dedalus data')
    parser.add_argument('--dt', type=float, 
                        help='dedalus data dt')
    parser.add_argument('--train_samples', type=int, default=100,
                        help='Number of training samples')
    parser.add_argument('--nTest', type=int, default=1,
                            help='Number of test samples')
    parser.add_argument('--folder', type=str, default=os.getcwd(),
                            help='Path to which FNO model inference is saved')
    parser.add_argument('--plotFile', type=str, default=None,
                        help='path to inference data file')
    parser.add_argument('--store_result',action='store_true',
                        help='store inference result to hdf5 file')
    parser.add_argument('--calc_loss',action='store_true',
                        help='calculate inference loss')
    args = parser.parse_args()
    
    main(**args.__dict__)