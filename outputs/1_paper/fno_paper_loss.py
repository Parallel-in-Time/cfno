import os
import sys
import numpy as np
import argparse 
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description='FNO Paper results',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--rootDir", help="directory containing loss_data")
parser.add_argument(
    "--plot_ablation",  action='store_true', help="plot obj, kernel, scheduler")
parser.add_argument(
    "--plot_run24",  action='store_true', help="plot final model loss")
parser.add_argument(
    "--plot_layer",  action='store_true', help="plot scaling layer ablation")
args = parser.parse_args()

root_dir = args.rootDir
start, end, step = 0, 1000, 9
logy = True
nRow, nCol = 2, 2
marker_style = '*-'

layer_width_plot_inds = [[[1, 58], [58, 60, 61]], [[5, 6, 18], [59, 19, 20]]]
layer_depth_plot_inds = [[[1, 58], [58, 5, 59]], [[60, 6, 19], [61, 18, 20]]]

model_ids = {
    1:  '1-Layer Linear (width = $d_v$)',
    57: '2-Layer Linear (width = $2d_v$)',
    58: '1-Layer Conv1D (width = $d_v$)',
    5:  '2-Layer Conv1D (width = $d_v$)',
    59: '4-Layer Conv1D (width = $d_v$)',
    60: '1-Layer Conv1D (width = $2d_v$)',
    6:  '2-Layer Conv1D (width = $2d_v$)',
    19: '4-Layer Conv1D (width = $2d_v$)',
    61: '1-Layer Conv1D (width = $4d_v$)',
    18: '2-Layer Conv1D (width = $4d_v$)',
    20: '4-Layer Conv1D (width = $4d_v$)'
}

obj = ['solution', 'update']
obj_id = [17, 1]
kernel = ['FNO', 'CFNO']
kernel_id = [1, 63]
sched = ['Cosine', 'StepLR']
sched_id = [24, 20]


color_map_ablation = {
    17: '#ff7f0e',   # solution
    1: '#9467bd',    # update
    63: '#1f77b4',   # CFNO
    24: '#9467bd',   # Cosine
    20: '#2ca02c'    # StepLR
}

color_map_width = {
    1: 'tab:blue',
    58: 'tab:orange', 5: 'tab:orange', 59: 'tab:orange',
    60: 'tab:green', 57: 'tab:green', 6: 'tab:green', 19: 'tab:green'
}

color_map_depth = {
    1: 'tab:blue',
    58: 'tab:orange', 60: 'tab:orange', 61: 'tab:orange',
    5: 'tab:green', 6: 'tab:green', 18: 'tab:green'
}

default_color = '#9467bd'


def plot_loss(ax, x, y, idLoss, model_id, label,color_map, log_scale=True):
    color = color_map.get(model_id, default_color)
    if log_scale:
        ax.semilogy(x, y, marker_style, c=color, markersize=4, label=label)
    else:
        ax.plot(x, y, marker_style, c=color, markersize=4, label=label)


def create_plot_grid(nRow, nCol, fig_title):
    fig, axs = plt.subplots(nRow, nCol, figsize=(8, 6), sharex=True, sharey=True)
    fig.supxlabel("Epoch", fontsize=12)
    fig.supylabel("Loss", fontsize=12)
    fig.suptitle(fig_title, fontsize=12)
    return fig, axs


def plot_group(fig, axs, plot_inds, color_map, plot_title, idLabel):
    for i in range(nRow):
        for j in range(nCol):
            le = True
            ax = axs[i][j]
            model_ids_to_plot = plot_inds[i][j]
            handles, labels = [], []
            for model_id in model_ids_to_plot:
                label = model_ids.get(model_id, f"Model {model_id}")
                loss_file = os.path.join(root_dir, f'losses_run{model_id}.txt')
                if not os.path.isfile(loss_file):
                    print(f"[Warning] File not found: {loss_file}")
                    continue
                data = np.loadtxt(loss_file).T
                nEpochs, trainLoss, validLoss, idTrain, idValid, grad = data[:6]
                x = nEpochs[start:end:step] + 1
                y = trainLoss[start:end:step]
                idLoss = idTrain[start:end:step]
                if plot_inds[0][0][0] == 1 and le:
                    ax.semilogy(x, idLoss, "--", c="black", label=f"idLoss ({idLabel})") 
                    le = False
                plot_loss(ax, x, y, idLoss, model_id, label, color_map, logy)
            handles1, labels1 = ax.get_legend_handles_labels()
            handles.extend(handles1)
            labels.extend(labels1)
            ax.legend(handles=handles, labels=labels, frameon=True, fontsize=10)
            ax.set_xlim(0, end + 50)
            ax.set_ylim(0.3, 5.0)
            ax.set_yticks([3.0, 1.0, 0.6, 0.5, 0.4])
            ax.set_yticklabels(['3.0', '1.0', '0.6', '0.5', '0.4'])
            ax.grid(True)
    fig.tight_layout()
    fig.savefig(f'{plot_title}.pdf')


def plot_ablations(fig, ax, ablate, ablate_id, ablate_char):
    for id in zip(ablate, ablate_id):
        loss_file = os.path.join(root_dir, f'losses_run{id[1]}.txt')
        data = np.loadtxt(loss_file).T
        nEpochs, trainLoss, validLoss, idTrain, idValid, grad = data[:6]
        if ablate_char == 'obj_1_17':
            color = color_map_ablation.get(id[1], default_color)
        else:
            color = 'black'
        if id[1] in [1,24]: 
            ax.semilogy(nEpochs[start:end:step]+1, idTrain[start:end:step], '--', c=color,label=f"idLoss (update)") 
        if id[1] == 17:
            ax.semilogy(nEpochs[start:end:step]+1, idTrain[start:end:step], '--', c=color, label=f"idLoss (solution)") 
        color = color_map_ablation.get(id[1], default_color)
        if logy:   
            ax.semilogy(nEpochs[start:end:step]+1, trainLoss[start:end:step], marker_style, c=color,  markersize=4, label=f"{id[0]}")
        else:
            ax.plot(nEpochs[start:end:step]+1, trainLoss[start:end:step], marker_style, c=color, markersize=4,label=f"{id[0]}")

        if id[1] == 1:
           ax.set_yticks([3.0, 1.0, 0.4,0.2, 0.002,0.00015 ],['3.0','1.0','0.4', '0.2', '2e-3', '1.5e-4']) # obj
        elif id[1] == 20:
           ax.set_yticks([3.0,1.0,0.8,0.6,0.4, 0.2],['3.0','1.0','0.8','0.6','0.4', '0.2']) 
        else:
           ax.set_yticks([3.0,1.0,0.8,0.6,0.5],['3.0','1.0','0.8','0.6','0.5'])  
    ax.legend(frameon=True)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_xlim(0, end+50)
    ax.grid(which="major")
    fig.tight_layout()
    fig.savefig(f'{ablate_char}.pdf')


def plot_run24(fig,ax):
    loss_file = os.path.join(root_dir, f'losses_run{24}.txt')
    data = np.loadtxt(loss_file).T
    nEpochs, trainLoss, validLoss, idTrain, idValid, grad = data[:6]
    ax.plot(nEpochs[start:end:step]+1, idTrain[start:end:step], "--", c="black",label=f"idLoss (update)") 
    ax.plot(nEpochs[start:end:step]+1, trainLoss[start:end:step], '-', c='#9467bd',label=f"trainLoss")
    ax.plot(nEpochs[start:end:step]+1, validLoss[start:end:step], '-', c='#e377c2',label=f"validLoss")
    ax.legend(frameon=True)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_yticks([1.5, 1.0, 0.5, 0.3,  0.2, 0.1],['1.5','1.0', '0.5', '0.3',  '0.2' ,'0.1'])
    ax.grid(which="major")
    fig.tight_layout()
    fig.savefig(f'run24.pdf')


if args.plot_layer:
    fig1, axs1 = create_plot_grid(nRow, nCol, "Layer Width Comparison")
    plot_group(fig1, axs1, layer_width_plot_inds, color_map_width, "layer_width", "update")

    fig2, axs2 = create_plot_grid(nRow, nCol, "Layer Depth Comparison")
    plot_group(fig2, axs2, layer_depth_plot_inds, color_map_depth, "layer_depth", "update")

if args.plot_ablation:
    fig3, ax3 = plt.subplots(1, 1,figsize=(6,4)) 
    plot_ablations(fig3, ax3, obj, obj_id, 'obj_1_17')

    fig4, ax4 = plt.subplots(1, 1,figsize=(6,4)) 
    plot_ablations(fig4, ax4, kernel, kernel_id, 'kernel_1_63')

    fig5, ax5 = plt.subplots(1, 1,figsize=(6,4)) 
    plot_ablations(fig5, ax5, sched,sched_id, 'obj_24_20')

if args.plot_run24:
    fig6, ax6 = plt.subplots(1, 1,figsize=(6,4)) 
    plot_run24(fig6, ax6)