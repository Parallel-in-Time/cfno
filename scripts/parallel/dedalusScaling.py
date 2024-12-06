from cfno.simulation.rbc2d import runSim, MPI_RANK, MPI_SIZE

infos = runSim(
    f"scaling_{MPI_SIZE}", tEnd=10, dtWrite=100, 
    writeSpaceDistr=True, logEvery=1000)
if MPI_RANK == 0:
    with open(f"infos_{MPI_SIZE}.txt", "w") as f:
        f.write(str(infos)+"\n")