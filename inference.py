import os
import sys
import functools
import operator
import h5py
import math
import copy
import scipy
import pickle
import scipy.io
import argparse
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from timeit import default_timer
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary    # TODO: is it really needed ?
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from utils import CudaMemoryDebugger, format_tensor_size, LpLoss, UnitGaussianNormalizer
from fno3d import FNO3d
from fno2d_recurrent import FNO2d

parser = argparse.ArgumentParser(description='Inference')
parser.add_argument('--model', type=str,
                    help=" Torch model state path")
parser.add_argument('--data_path', type=str,
                    help='path to data')
parser.add_argument('--dim', type=str,default="FNO2D",
                    help="FNO2D+recurrent time or FNO3D")
parser.add_argument('--modes', type=int, default=12,
                    help="Fourier modes")
parser.add_argument('--width', type=int, default=20,
                    help="Width of layer")
parser.add_argument('--batch_size', type=int, default=5,
                    help="Batch size")
parser.add_argument('--folder', type=str,default=os.getcwd(),
                        help='path to which FNO model inference is saved')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")
torch.cuda.empty_cache()
memory = CudaMemoryDebugger(print_mem=True)


## config
modes = args.modes
width = args.width
batch_size = args.batch_size

gridx = 4*256
gridz = 64

xStep = 1
zStep = 1
tStep = 1

start_index = 500
T_in = 10
T = 10

myloss = LpLoss(size_average=False)
## load data
ntrain = 100
ntest = 50

data_path = args.data_path
reader = h5py.File(data_path, mode="r")
train_a = torch.tensor(reader['train'][:ntrain, ::xStep, ::zStep, start_index: start_index+T_in],dtype=torch.float)
train_u = torch.tensor(reader['train'][:ntrain, ::xStep, ::zStep, start_index+T_in:T+start_index+T_in], dtype=torch.float)

test_a = torch.tensor(reader['test'][:ntest, ::xStep, ::zStep, start_index: start_index+T_in],dtype=torch.float)
test_u = torch.tensor(reader['test'][:ntest, ::xStep, ::zStep, start_index+T_in:T+start_index+T_in],dtype=torch.float)

fno_path = Path(f'{args.folder}/rbc_fno_3d_N{ntest}_m{modes}_w{width}_bs{batch_size}_inference')
fno_path.mkdir(parents=True, exist_ok=True)

if args.dim == 'FNO3D':
    a_normalizer = UnitGaussianNormalizer(train_a)
    test_a = a_normalizer.encode(test_a)
    y_normalizer = UnitGaussianNormalizer(train_u)
    test_u = y_normalizer.encode(test_u)
    test_a = test_a.reshape(nval, gridx, gridz, 1, T_in).repeat([1,1,1,T,1])
    model = FNO3d(modes, modes, modes, width).to(device)
else:
    model = FNO2d(modes, modes, width).to(device)

model.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))   
print(f"Test data:{test_u.shape}")
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

pred = torch.zeros(test_u.shape)
index = 0
inputs = []
outputs = []
predictions = []
with torch.no_grad():
    for step, (xx, yy) in enumerate(test_loader):
        test_l2 = 0
        xx, yy = xx.to(device), yy.to(device)
        if args.dim == 'FNO3D':
            out = model(xx)
            out = y_normalizer.decode(out)
            test_l2 += myloss(out.view(1, -1), yy.view(1, -1)).item()
            pred[step] = out
        else:
            for t in range(0, T, tStep):
                y = yy[..., t:t + tStep]
                im = model(xx)

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., tStep:], im), dim=-1)
            test_l2 += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

        inputs.append(xx)
        outputs.append(yy)
        predictions.append(pred)
        print(f"index:{step} loss:{test_l2}")

with h5py.File(f'{fno_path}/result.h5', "w") as data:
    data['input'] = inputs
    data['output'] = outputs
    data['prediction'] = predictions
data.close()