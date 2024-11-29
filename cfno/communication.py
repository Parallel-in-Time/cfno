import os 
import torch
import torch.distributed as dist

class Communicator:
    """
    Communicator for distributed training 
    """
    
    def __init__(self,
                 gpus_per_node:int=4,
                 rank:int=0,
    ):
        """
        Constructor for communication constructor

        Args:
            gpus_per_node (int): number of GPUs per node. Default is 4.
            rank (int): rank of worker in worker group
        """

        device_count = torch.cuda.device_count()
        
        if dist.is_initialized():
            if rank == 0:
                print(
                    "torch distributed is already initialized, "
                    "skipping initialization ...",
                    flush=True,
                )
        else:
            if rank == 0:
                print("> initializing torch distributed ...", flush=True)
            # Manually set the device ids.
            if device_count > 0:
                self.local_rank = int(os.getenv('LOCAL_RANK', '0'))
                torch.cuda.set_device(self.local_rank)
                self.device_id = torch.device(f'cuda:{self.local_rank}')
            else:
                self.device_id = None

            # Call the init process
            init_process_group_kwargs = {
                'backend' : "nccl",
                'world_size': device_count,
                'rank': rank,
            }

            dist.init_process_group(**init_process_group_kwargs)
            
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        # self.dp_group = list(range(self.world_size))
        self.local_process_group_size = gpus_per_node
        
        torch.cuda.empty_cache()
        print(f"[{os.getpid()}, {self.device_id}] world_size = {self.world_size}, "
        + f"rank = {self.rank}, backend={dist.get_backend()}"
        )
            
    def cleanup_communicator():
        """
        Destroy torch process group
        """
        dist.destroy_process_group()
          
    def allreduce(self, tensor, async_op: bool=False, op=dist.ReduceOp.SUM):
        """
        Allreduce Pytorch tensors using NCCL 

        Args:
            tensor (torch.tensor): PyTorch tensor to all-reduce
            async_op (bool, optional): to make async operation
            op : torch collective operation
        """
        return dist.all_reduce(tensor, async_op=async_op)
        

def get_local_rank():
    """
    Function to get local_rank of process group
    """
    if not dist.is_initialized():
        return 0
    else:
        return dist.get_rank()
    
def get_world_size():
    """
    Function to get world size of process group
    """
    if not dist.is_initialized():
        return 1
    else:
        return dist.get_world_size()