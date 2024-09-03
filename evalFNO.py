"""
Starting tentative simplify the FNO evaluation from the inference.py script
"""

import os
import numpy as np
from inference import model_inference, time_extract

strat = "FNO2D_RBC2D_strategy2"
model = "model_nx256_nz64_dt1e_3_tin1"
file  = "model_checkpoint_2999.pt"
path  = f"../model_archive/{strat}/{model}/{file}"

if not os.path.isfile(path):
    raise SystemError(
        "need a clone of model_archive in the same folder as neural_operators")

start = 500
stop = 820
step_time = 10
T_in = 1
T = 1

time_file = "dedalus/run_sdc_M1/run_init_s1.h5"

class args:
    single_data_path = None
    dim = "FNO2D"
    # TODO : needs more arguments

for iteration, start_index in enumerate(range(start, stop, step_time)):
    start_index_org = 0 + start_index
    time_in, _ = time_extract(time_file, start_index_org, T_in, T)

    inputs, outputs, predictions = model_inference(args)
    print(f"Model Inference: Input{inputs.shape}, Output{outputs.shape}, Prediction{predictions.shape}")

    batches = predictions.shape[0]
    batchsize = predictions.shape[1]
    batch_num = np.random.randint(0, batches)
    sample = np.random.randint(0, batchsize)

    # TODO : rework from here
    if args.dim == "FNO3D":
        ux, uz, b_in, p_in = extract(inputs[batch_num, sample, :, :, 0, :], gridx//4, gridz, T_in)
    else:
        ux, uz, b_in, p_in = extract(inputs[batch_num, sample, :, :, :], gridx//4, gridz, T_in)

    vx1, vz1, b_out1, p_out1 = extract(predictions[batch_num, sample, :, :, :], gridx//4, gridz,T)
    vx, vz, b_out, p_out = extract(outputs[batch_num, sample, :, :, :], gridx//4, gridz, T)
