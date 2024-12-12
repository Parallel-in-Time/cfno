import os
import numpy as np
from datetime import datetime

import dedalus.public as d3
from mpi4py import MPI
from pySDC.playgrounds.dedalus.sdc import SpectralDeferredCorrectionIMEX

COMM_WORLD = MPI.COMM_WORLD
MPI_SIZE = COMM_WORLD.Get_size()
MPI_RANK = COMM_WORLD.Get_rank()

def runSim(dirName, Rayleigh=1e6, resFactor=1, baseDt=1e-2/2, seed=999,
    tBeg=0, tEnd=150, dtWrite=0.1, useSDC=False,
    writeVort=False, writeFull=False, initFields=None,
    writeSpaceDistr=False, logEvery=100, distrMesh=None):
    """
    Run RBC simulation in a given folder.

    Parameters
    ----------
    dirName: str
        Name of directory where to store snapshots and run infos
    Rayleigh: float
        Rayleigh number.
    resFactor: int
        Resolution factor, considering a base space grid size of (256,64).
    baseDt: float
        Base time-step for the base space resolution. The default is 1e-2/2.
    seed: int, optional
        Seed for the random noise in the initial solution. The default is 999.
    tBeg: float
        Simulation starting time (default is 0)
    tEnd: float
        Simulation end time
    dtWrite: float
        Snapshot save interval
    useSDC: boolean, optional
        Use SDC timestepper
    writeVort: boolean, optional
        Write vorticity to snapshot
    writeFull: boolean, optional
        Write Tau variables to snapshot
    initFields: dictionary, optional
        Initial conditions
    writeSpaceDistr: bool, optional
        Write into a file the space parallel distribution from dedalus
    """
    # Parameters
    Lx, Lz = 4, 1
    Nx, Nz = 256*resFactor, 64*resFactor
    timestep = baseDt/resFactor

    nSteps = round(float(tEnd-tBeg)/timestep, ndigits=3)
    if float(tEnd-tBeg) != round(nSteps*timestep, ndigits=3):
        raise ValueError(
            f"tEnd ({tEnd}) is not divisible by timestep ({timestep}) (nSteps={nSteps})")
    nSteps = int(nSteps)
    infos = {
        "nSteps": nSteps+1,
        "nDOF": Nx*Nz
    }
    if os.path.isfile(f"{dirName}/01_finalized.txt"):
        if MPI_RANK == 0:
            print(" -- simulation already finalized, skipping !")
        return infos, None

    def log(msg):
        if MPI_RANK == 0:
            with open(f"{dirName}/simu.log", "a") as f:
                f.write(f"{dirName} -- ")
                f.write(datetime.now().strftime("%d/%m/%Y  %H:%M:%S"))
                f.write(f", MPI rank {MPI_RANK} ({MPI_SIZE})")
                f.write(f" : {msg}\n")

    Prandtl = 1
    dealias = 3/2
    stop_sim_time = tEnd
    timestepper = SpectralDeferredCorrectionIMEX if useSDC else d3.RK443
    dtype = np.float64

    os.makedirs(dirName, exist_ok=True)
    with open(f"{dirName}/00_infoSimu.txt", "w") as f:
        f.write(f"Rayleigh : {Rayleigh:1.2e}\n")
        f.write(f"Seed : {seed}\n")
        f.write(f"Nx, Nz : {int(Nx)}, {int(Nz)}\n")
        f.write(f"dt : {timestep:1.2e}\n")
        f.write(f"tEnd : {stop_sim_time}\n")
        f.write(f"dtWrite : {dtWrite}\n")

    # Bases
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=dtype, mesh=distrMesh)
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
    zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

    # Fields
    p = dist.Field(name='p', bases=(xbasis,zbasis))
    b = dist.Field(name='b', bases=(xbasis,zbasis))
    u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
    tau_p = dist.Field(name='tau_p')
    tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
    tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

    # Substitutions
    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)
    x, z = dist.local_grids(xbasis, zbasis)
    ex, ez = coords.unit_vector_fields(dist)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
    grad_b = d3.grad(b) + ez*lift(tau_b1) # First-order reduction

    if writeSpaceDistr:
        print(f"Rank {MPI_RANK}/{MPI_SIZE} :\n"
              f"\tx: {x.shape}, [{x.min(initial=np.inf)}, {x.max(initial=-np.inf)}]\n"
              f"\tz: {z.shape}, [{z.min(initial=np.inf)}, {z.max(initial=-np.inf)}]\n"
              f"cpu: {os.sched_getaffinity(0)}")

    # Problem
    # First-order form: "div(f)" becomes "trace(grad_f)"
    # First-order form: "lap(f)" becomes "div(grad_f)"
    problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
    problem.add_equation("trace(grad_u) + tau_p = 0")
    problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
    problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)")
    problem.add_equation("b(z=0) = Lz")
    problem.add_equation("u(z=0) = 0")
    problem.add_equation("b(z=Lz) = 0")
    problem.add_equation("u(z=Lz) = 0")
    problem.add_equation("integ(p) = 0") # Pressure gauge

    # Solver
    solver = problem.build_solver(timestepper)
    solver.sim_time = tBeg
    solver.stop_sim_time = stop_sim_time

    # Initial conditions
    if initFields is None:
        b.fill_random('g', seed=seed, distribution='normal', scale=1e-3) # Random noise
        b['g'] *= z * (Lz - z) # Damp noise at walls
        b['g'] += Lz - z # Add linear background
    else:
        fields = [
            (b, "buoyancy"), (p, "pressure"), (u, "velocity"),
            *[(f, f.name) for f in [tau_p, tau_b1, tau_b2, tau_u1, tau_u2]]
            ]
        for field, name in fields:
            localSlices = (slice(None),) * len(field.tensorsig) \
                + dist.grid_layout.slices(field.domain, field.scales)
            field['g'] = initFields[name][-1][localSlices]

    # Analysis
    iterWrite = dtWrite/timestep
    if int(iterWrite) != round(iterWrite, ndigits=3):
        raise ValueError(
            f"dtWrite ({dtWrite}) is not divisible by timestep ({timestep}) : {iterWrite}")
    iterWrite = int(iterWrite)
    snapshots = solver.evaluator.add_file_handler(
        dirName, sim_dt=dtWrite, max_writes=stop_sim_time/timestep)
    snapshots.add_task(u, name='velocity')
    snapshots.add_task(b, name='buoyancy')
    snapshots.add_task(p, name='pressure')
    if writeFull:
        snapshots.add_task(tau_p, name='tau_p')
        snapshots.add_task(tau_b1, name='tau_b1')
        snapshots.add_task(tau_b2, name='tau_b2')
        snapshots.add_task(tau_u1, name='tau_u1')
        snapshots.add_task(tau_u2, name='tau_u2')
    if writeVort:
        snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

    # Main loop
    if nSteps == 0:
        return
    try:
        log('Starting main loop')
        t0 = MPI.Wtime()
        for _ in range(nSteps+1): # need to do one more step to write last solution ...
            solver.step(timestep)
            if (solver.iteration-1) % logEvery == 0:
                log(f'Iteration={solver.iteration}, Time={solver.sim_time}, dt={timestep}')
        t1 = MPI.Wtime()
        infos["tComp"] = t1-t0
        if MPI_RANK == 0:
            with open(f"{dirName}/01_finalized.txt", "w") as f:
                f.write("Done !")
    except:
        log('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()

    return infos, solver
