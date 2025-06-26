import os, sys
sys.path.insert(2, os.getcwd())
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from cfno.communication import Communicator
from cfno.models.cfno2d import CFNO2D
from cfno.losses import VectormNormLoss

def cleanup():
     dist.destroy_process_group()
     
def run_train(gpus_per_node, rank, model):
    print(f"Testing DDP training on rank {rank}")
    
    communicator = Communicator(gpus_per_node, rank)
    device = communicator.device
    
    model = CFNO2D(**model).to(device)
    ddp_model = DDP(model, device_ids=[communicator.local_rank])
    
    loss_fn = VectormNormLoss()
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=0.00039)
    labels = torch.randn(5, 4, 256, 64).to(device)
    
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(5, 4, 256, 64))
    loss_fn(outputs, labels).backward()
    optimizer.step()
    
    cleanup()
    
    
if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    model = {"da": 4, "du": 4, "dv": 6, "kX":12, "kY":12,
             "nLayers":4, "non_linearity": "gelu", "forceFFT": False,
             "bias": True}
    if n_gpus < 2:
        print(f"Requires at least 2 GPUs to run, but got {n_gpus}.")
    else:
        rank = int(os.getenv('RANK', '0'))
        run_train(n_gpus, rank, model)
  
      