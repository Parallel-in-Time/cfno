import copy
import json
import os
import sys

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from problems import Darcy, WaveEquation, RBC2D
from problems import default_param, default_train_params, RBC_param, RBC_train_param

if len(sys.argv) == 2:

    training_properties = {}
    fno_architecture = {}
    
    #   "which_example" can be 
    #   wave           : Wave equation
    #   darcy          : Darcy Flow
    #   RBC2D          : Rayleigh-Bénard convection 2D
    
    which_example = sys.argv[1]
    if which_example == "wave" or which_example == "darcy":
        training_properties = default_train_params(training_properties)
        fno_architecture = default_param(fno_architecture)
    elif which_example == "RBC2D":
        training_properties = RBC_train_param(training_properties)
        fno_architecture = RBC_param(fno_architecture)
    else:
        raise ValueError("the variable which_example has to be either wave or darcy or RBC2D")

    # Save the models here:
    folder = "TrainedModels/"+"FNO_"+which_example+"_tmp1"

else:
    folder = sys.argv[1]
    training_properties = json.loads(sys.argv[2].replace("\'", "\""))
    fno_architecture = json.loads(sys.argv[3].replace("\'", "\""))
    which_example = sys.argv[4]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir=folder+"/tensorboard")

learning_rate = training_properties["learning_rate"]
epochs = training_properties["epochs"]
batch_size = training_properties["batch_size"]
weight_decay = training_properties["weight_decay"]
scheduler_step = training_properties["scheduler_step"]
scheduler_gamma = training_properties["scheduler_gamma"]
training_samples = training_properties["training_samples"]
p = training_properties["exp"]


if which_example == "wave":
    example = WaveEquation(fno_architecture, device, batch_size,training_samples)
elif which_example == "darcy":
    example = Darcy(fno_architecture, device, batch_size,training_samples)
elif which_example == "RBC2D":
    example = RBC2D(fno_architecture, device, batch_size,training_samples)
else:
    raise ValueError("the variable which_example has to be either wave or darcy or RBC2D")

if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)

df = pd.DataFrame.from_dict([training_properties]).T
df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='w')
df = pd.DataFrame.from_dict([fno_architecture]).T
df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')

model = example.model
n_params = model.print_size()
train_loader = example.train_loader
val_loader = example.val_loader

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)


if p == 1:
    loss = torch.nn.SmoothL1Loss()  # https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
elif p == 2:
    loss = torch.nn.MSELoss()
    
best_model_testing_error = 300
threshold = int(0.25 * epochs)
counter = 0

for epoch in range(epochs):
    with tqdm(unit="batch", disable=False) as tepoch:
        model.train()
        tepoch.set_description(f"Epoch {epoch}")
        train_mse = 0.0
        # running_relative_train_mse = 0.0
        for step, (input_batch, output_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            input_batch = input_batch.to(device)
            output_batch = output_batch.to(device)
            output_pred_batch = model(input_batch)

            loss_f = loss(output_pred_batch, output_batch)/ loss(torch.zeros_like(output_batch).to(device), output_batch)

            loss_f.backward()
            optimizer.step()
            train_mse = train_mse * step / (step + 1) + loss_f.item() / (step + 1)   # taking norm
            tepoch.set_postfix({'Batch': step + 1, 'Train loss (in progress)': train_mse})
           
            
        writer.add_scalar("train_loss/train_loss", train_mse, epoch)

        with torch.no_grad():
            model.eval()
            val_relative_l1 = 0.0
            train_relative_l1 = 0.0

            for step, (input_batch, output_batch) in enumerate(val_loader):
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)
                output_pred_batch = model(input_batch)
                
                loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100
                val_relative_l1 += loss_f.item()
            val_relative_l1 /= len(val_loader)
            
            for step, (input_batch, output_batch) in enumerate(train_loader):
                    input_batch = input_batch.to(device)
                    output_batch = output_batch.to(device)
                    output_pred_batch = model(input_batch)
                    
                    loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100
                    train_relative_l1 += loss_f.item()
            train_relative_l1 /= len(train_loader)
            
            writer.add_scalar("train_loss/train_l1_loss_rel", train_relative_l1, epoch)
            writer.add_scalar("val_loss/val_l1_loss_rel", val_relative_l1, epoch)

            if val_relative_l1 < best_model_testing_error:
                best_model_testing_error = val_relative_l1
                best_model = copy.deepcopy(model)
                torch.save(best_model, folder + "/model.pkl")
                writer.add_scalar("val_loss/Best Relative Testing Error", best_model_testing_error, epoch)
                counter = 0
            else:
                counter +=1

        tepoch.set_postfix({'Train loss': train_mse, "Relative Train L1 loss": train_relative_l1, "Relative Val L1 loss": val_relative_l1})
        tepoch.close()
        
        with open(folder + '/errors.txt', 'w') as file:
            file.write("Training Error: " + str(train_mse) + "\n")
            file.write("Best Testing Error: " + str(best_model_testing_error) + "\n")
            file.write("Current Epoch: " + str(epoch) + "\n")
            file.write("Params: " + str(n_params) + "\n")
        scheduler.step()
    
    if counter > threshold:
        print("Early Stopping since best_model_testing_error:{best_model_testing_error} < val_relative_l1:{val_relative_l1} \
                in the given threshold:{threshold}")
        break