import torch
from problems_darcy_wave import Darcy, WaveEquation, default_param, default_train_params
from problems_rbc import RBC2D, RBC_param, RBC_train_param
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='FNO Inference')
parser.add_argument('--model_state_path', type=str,
                    help=" Torch model state path")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network_properties = {}
training_properties = {}
fno_architecture = RBC_param(network_properties)
training_properties = RBC_train_param(training_properties)
learning_rate = training_properties["learning_rate"]
epochs = training_properties["epochs"]
batch_size = training_properties["batch_size"]
weight_decay = training_properties["weight_decay"]
scheduler_step = training_properties["scheduler_step"]
scheduler_gamma = training_properties["scheduler_gamma"]
training_samples = training_properties["training_samples"]
p = training_properties["exp"]
problem = RBC2D(fno_architecture, device, batch_size,training_samples)
model_state_path  = args.model_state_path


index = 5
model = problem.model
model.load_state_dict(torch.load(model_state_path,  map_location=lambda storage, loc: storage))
for step, (input_batch, output_batch) in enumerate(problem.test_loader):
            output_pred_batch = model(input_batch)
            print("input, output, prediction = ",input_batch.shape, output_batch.shape, output_pred_batch.shape)
            for i in range(index,index+1):
                x = input_batch[i]
                y = output_batch[i]
                z = output_pred_batch[i].detach().numpy()
                # print(x.shape,y.shape, z.shape)
                print(np.linalg.norm(y-z))
            break
