import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def cross_section_error_plot( vx:np.ndarray, vx_pred:np.ndarray,
                    vy:np.ndarray, vy_pred:np.ndarray,
                    b_out:np.ndarray, b_pred:np.ndarray,
                    p_out:np.ndarray, p_pred:np.ndarray, 
                    time_in:list, time_out:list, 
                    dim:str, fno_path:str,
                    gridx:int, gridy:int,
                    rayleigh:float, prandtl:float,
                    plot_input:bool=False, 
                    **kwargs):
    """
    Plotting cross-sections of velocity, buoyancy and pressure data 
    on grid with error bars 

    Args:
        vx (np.ndarray): Dedalus velocity x-component output
        vx_pred (np.ndarray): FNO model velocty x-component output
        vy (np.ndarray): Dedalus velocity y-component output
        vy_pred (np.ndarray): FNO model velocty y-component output
        b_out (np.ndarray): Dedalus buoyancy output
        b_pred (np.ndarray): FNO model buoyancy output
        p_out (np.ndarray): Dedalus pressure output
        p_pred (np.ndarray): FNO model pressure output
        time_in (list): list of input simulation times
        time_out (list): list of output simulation times
        dim (str):  FNO2D or FNO3D strategy
        fno_path (str): path to store plots
        gridx (int): x grid size
        gridy (int): y grid size
        rayleigh (float): Rayleigh Number
        prandtl (float): Prandtl number 
        
    Optional:
        plot_input (bool): plot input metrics. Default is False.
        ux (np.ndarray): Dedalus velocity x-component input 
        uy (np.ndarray): Dedalus velocity y-component input
        b_in (np.ndarray): Dedalus buoyancy input
        p_in (np.ndarray):  Dedalus pressure input
    """
    for t in range(len(time_out)):
        row = 2
        col = 4
        xStep = 30
        yStep = 30
        x = np.arange(0,gridx,xStep)
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
        ax3.plot(x,p_out[::xStep,::yStep,t],marker ='o',color='g',label="ded-p")
        ax3.plot(x,p_pred[::xStep,::yStep,t],marker ='o',color='r',linestyle='--',label="fno-p")
        # ax3.set_ylabel("Y grid")
        ax3.grid()

        ax4.set_title(fr'Buoyancy: $b(x,z)$')
        ax4.plot(x,b_out[::xStep,::yStep,t],marker ='o',color='g',label="ded-b")
        ax4.plot(x,b_pred[::xStep,::yStep,t],marker ='o',linestyle='--',color='r',label="fno-b")
        # ax4.set_ylabel("Y grid")
        ax4.grid()
        
        if plot_input and kwargs:
            ux = kwargs["ux"]
            uy = kwargs["uy"]
            p_in = kwargs["p_in"]
            b_in = kwargs["b_in"]
            ax1.plot(x,ux[::xStep,::yStep,t],color='b',marker ='o',label="ux")
            ax2.plot(x,uy[::xStep,::yStep,t],color='b',marker ='o',label="uy")
            ax3.plot(x,p_in[::xStep,::yStep,t],color='b',marker ='o',label="p_in")
            ax4.plot(x,b_in[::xStep,::yStep,t],marker ='o',color='b',label="b_in")

        ax5.errorbar(x, np.average(vx[::xStep,::yStep,t],axis=1), yerr=np.average(np.abs(vx_pred[::xStep,::yStep,t]-vx[::xStep,::yStep,t]), axis=1), marker='o',color='purple',capsize=3, markersize=6,linestyle='none')
        ax5.set_ylabel(r"$\overline{|vx_{ded}-vx_{fno}|}_{z}$")
        ax5.set_xlabel("X Grid")

        ax6.errorbar(x, np.average(vy[::xStep,::yStep,t],axis=1), yerr=np.average(np.abs(vy_pred[::xStep,::yStep,t]-vy[::xStep,::yStep,t]), axis=1), marker='o',color='purple',capsize=3, markersize=6,linestyle='none')
        ax6.set_ylabel(r"$\overline{|vz_{ded}-vz_{fno}|}_{z}$")
        ax6.set_xlabel("X Grid")

        ax7.errorbar(x, np.average(p_out[::xStep,::yStep,t],axis=1), yerr=np.average(np.abs(p_pred[::xStep,::yStep,t]-p_out[::xStep,::yStep,t]), axis=1), marker='o',color='purple',capsize=3, markersize=6,linestyle='none')
        ax7.set_ylabel(r"$\overline{|p_{ded}-p_{fno}|}_{z}$")
        ax7.set_xlabel("X Grid")

        ax8.errorbar(x, np.average(b_out[::xStep,::yStep,t],axis=1), yerr=np.average(np.abs(b_pred[::xStep,::yStep,t]-b_out[::xStep,::yStep,t]), axis=1), marker='o',color='purple',capsize=3, markersize=6,linestyle='none')
        ax8.set_ylabel(r"$\overline{|b_{ded}-b_{fno}|}_{z}$")
        ax8.set_xlabel("X Grid")

        fig.suptitle(f'RBC-2D with {gridx}'+r'$\times$'+f'{gridy} grid and Ra={rayleigh} and Pr={prandtl} using {dim}')  
        if len(time_in) > 1:
            inp_patch = Line2D([0], [0], label=f'Input at t={np.round(time_in[0],4)}:{np.round(time_in[-1],4)}',marker='o', color='b')
        else:
            inp_patch = Line2D([0], [0], label=f'Input at t={np.round(time_in[0],4)}',marker='o', color='b')
        ded_patch = Line2D([0], [0], label=f'Dedalus at t={np.round(time_out[t],4)}',marker='o', color='g')
        fno_patch = Line2D([0], [0], label=f'FNO at t={np.round(time_out[t],4)}',marker='o', linestyle='--', color='r')
       
        fig.legend(handles=[inp_patch, ded_patch, fno_patch], loc="upper right")
        # fig.tight_layout()
        fig.show()
        fig.savefig(f"{fno_path}/{dim}_NX{gridx}_NY{gridy}_{np.round(time_out[t],4)}.png")
        
def cross_section_plots(infFile:str, 
                       gridx:int, gridy:int,
                       dim:str, fno_path:str,
                       rayleigh:float, prandtl:float,
                       T_in:int=1, T:int=1,
                       plot_input:bool=False,):
    """
    Funtion to plot cross-sections of velocity, 
    buoyancy and pressure from an hdf5 file

    Args:
        infFile (str): path to inference hdf5 file
        gridx (int): x grid size
        gridy (int): y grid size
        dim (str): FNO2D or FNO3D strategy
        fno_path (str): path to store plots
        rayleigh (float): Rayleigh Number
        prandtl (float): Prandtl number 
        T_in (int, optional): number of input timesteps. Defaults to 1.
        T (int, optional): number of output timesteps. Defaults to 1.
        plot_input (bool): to extract and plot input data 
        
    """
    with h5py.File(infFile, "r") as data:
        for iteration in range(len(data.keys())):
            time_in = []
            time_out = []
            vx = np.zeros((gridx, gridy, T))
            vy = np.zeros((gridx, gridy, T))
            vx_pred = np.zeros((gridx, gridy, T))
            vy_pred = np.zeros((gridx, gridy, T))
            p_out = np.zeros((gridx, gridy, T))
            p_pred = np.zeros((gridx, gridy, T))
            b_out = np.zeros((gridx, gridy, T))
            b_pred = np.zeros((gridx, gridy, T)) 
            
            if plot_input:
                ux = np.zeros((gridx, gridy, T_in))
                uy = np.zeros((gridx, gridy, T_in))
                p_in = np.zeros((gridx, gridy, T_in))
                b_in = np.zeros((gridx, gridy, T_in))
                
            for index_in in range(T_in):
                time_in.append(data[f'inference_{iteration}/scales/sim_timein_{index_in}'])
                if plot_input:
                    ux[:,:,index_in] = data[f'inference_{iteration}/tasks/input/velocity_{index_in}'][0,:]
                    uy[:,:,index_in] = data[f'inference_{iteration}/tasks/input/velocity_{index_in}'][1,:]
                    b_in[:,:,index_in] = data[f'inference_{iteration}/tasks/input/buoyancy_{index_in}'][:]
                    p_in[:,:,index_in] = data[f'inference_{iteration}/tasks/input/pressure_{index_in}'][:]
                    
            for index_out in range(T):
                time_out.append(data[f'inference_{iteration}/scales/sim_timeout_{index_out}'])
                vx[:,:,index_out] = data[f'inference_{iteration}/tasks/output/velocity_{index_out}'][0,:]
                vy[:,:,index_out] = data[f'inference_{iteration}/tasks/output/velocity_{index_out}'][1,:]
                b_out[:,:,index_out] = data[f'inference_{iteration}/tasks/output/buoyancy_{index_out}'][:]
                p_out[:,:,index_out] = data[f'inference_{iteration}/tasks/output/pressure_{index_out}'][:]
                vx_pred[:,:,index_out] = data[f'inference_{iteration}/tasks/model_output/velocity_{index_out}'][0,:]
                vy_pred[:,:,index_out] = data[f'inference_{iteration}/tasks/model_output/velocity_{index_out}'][1,:]
                b_pred[:,:,index_out] = data[f'inference_{iteration}/tasks/model_output/buoyancy_{index_out}'][:]
                p_pred[:,:,index_out] = data[f'inference_{iteration}/tasks/model_output/pressure_{index_out}'][:]

            # print(vx.shape, vy.shape, b_pred.shape, p_pred.shape)
            cross_section_error_plot(vx, vx_pred,vy, vy_pred, b_out, b_pred, p_out, p_pred,
                time_in, time_out, dim, fno_path, gridx, gridy, rayleigh, prandtl, plot_input)
