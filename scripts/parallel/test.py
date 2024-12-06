import torch as th
from mpi4py import MPI

th.manual_seed(0)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def cudaMemInfo():
    free, total = th.cuda.mem_get_info(0)
    used = th.cuda.max_memory_allocated(0)
    return f"used:{used}, free:{free}, total:{total}"

print(f"Process {rank} (before model): {cudaMemInfo()}")

model = th.nn.Linear(10, 2).cuda()
comm.Barrier()
print(f"Process {rank} (after model): {cudaMemInfo()}")

# All ranks use the shared model
input_data = th.randn(10, 10).cuda()
print(f"Process {rank} (after input): {cudaMemInfo()}")
with th.no_grad():
    output = model(input_data)
print(f"Process {rank} (after inference): {cudaMemInfo()}")
print("-"*80)