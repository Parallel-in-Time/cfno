from cfno.simulation.rbc2d import runSim, MPI_RANK, MPI_SIZE

tEnd = 1

infos, solver, b = runSim(
    f"scaling_{MPI_SIZE}",
    tEnd=tEnd, dtWrite=2*tEnd, writeSpaceDistr=True, logEvery=10000,
    distrMesh=None)
if MPI_RANK == 0:
    with open(f"infos_{MPI_SIZE:03d}.txt", "w") as f:
        f.write(str(infos)+"\n")
