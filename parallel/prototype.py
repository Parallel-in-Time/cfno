#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mpi4py import MPI
import multiprocessing
import psutil

def mpi_function(rank, size):
    """
    A function that uses MPI parallelization. Each process will print its rank,
    the total number of processes, and the CPU core it's running on.
    """
    # Get the CPU core number (using psutil for cross-platform compatibility)
    core_num = psutil.Process().cpu_num()

    # Print the information
    print(f"MPI Rank {rank}/{size} is running on CPU core {core_num}")

def run_mpi():
    """
    This function initializes the MPI environment and runs the mpi_function in
    parallel across all MPI processes.
    """
    comm = MPI.COMM_WORLD  # Initialize MPI communicator
    rank = comm.Get_rank()  # Get the rank of the process
    size = comm.Get_size()  # Get the total number of processes

    # Call the MPI function
    mpi_function(rank, size)

if __name__ == "__main__":
    # Number of processes to spawn using multiprocessing
    num_processes = 4

    # Use multiprocessing to spawn multiple processes
    processes = []
    for _ in range(num_processes):
        p = multiprocessing.Process(target=run_mpi)
        processes.append(p)
        p.start()

    # Ensure all processes have completed
    for p in processes:
        p.join()
