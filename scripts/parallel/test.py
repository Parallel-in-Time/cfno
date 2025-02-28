from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

mesh = np.array([size], dtype=int)
cartComm = comm.Create_cart(mesh)
print(f"rank {rank}({size}) : coords={cartComm.coords}")
