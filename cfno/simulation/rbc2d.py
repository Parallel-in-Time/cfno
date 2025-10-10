import os
import socket
import numpy as np
from datetime import datetime
from time import sleep

import dedalus.public as d3
from mpi4py import MPI
from pySDC.playgrounds.dedalus.sdc import SpectralDeferredCorrectionIMEX, SDCIMEX_MPI, SDCIMEX_MPI2

COMM_WORLD = MPI.COMM_WORLD
MPI_SIZE = COMM_WORLD.Get_size()
MPI_RANK = COMM_WORLD.Get_rank()

def runSim(dirName, Rayleigh=1e7, resFactor=1, baseDt=1e-2/2, seed=999,
    tBeg=0, tEnd=150, dtWrite=0.1, useSDC=False,
    writeVort=False, writeFull=False, initFields=None,
    writeSpaceDistr=False, logEvery=100, distrMesh=None,
    timeParallel=False, groupTimeProcs=False, useTimePar2=False):
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
    # Lx, Lz = 1.5, 1
    # Nx, Nz = 96*resFactor, 64*resFactor
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
    # Prandtl = 0.7
    dealias = 3/2
    stop_sim_time = tEnd
    timestepper = SpectralDeferredCorrectionIMEX if useSDC else d3.RK443
    dtype = np.float64

    if timeParallel:
        assert useSDC, "cannot use time-parallel without SDC"
        _, sComm, _ = SDCIMEX_MPI.initSpaceTimeComms(groupTime=groupTimeProcs)
        timestepper = SDCIMEX_MPI2 if useTimePar2 else SDCIMEX_MPI
        if MPI_RANK == 0:
            print(" -- using timeParallel implementation" + " 2" if useTimePar2 else "")
    else:
        sComm = COMM_WORLD

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
    distr = d3.Distributor(coords, dtype=dtype, mesh=distrMesh, comm=sComm)
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
    zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

    # Fields
    p = distr.Field(name='p', bases=(xbasis,zbasis))
    b = distr.Field(name='b', bases=(xbasis,zbasis))
    u = distr.VectorField(coords, name='u', bases=(xbasis,zbasis))
    tau_p = distr.Field(name='tau_p')
    tau_b1 = distr.Field(name='tau_b1', bases=xbasis)
    tau_b2 = distr.Field(name='tau_b2', bases=xbasis)
    tau_u1 = distr.VectorField(coords, name='tau_u1', bases=xbasis)
    tau_u2 = distr.VectorField(coords, name='tau_u2', bases=xbasis)

    # Substitutions
    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)
    x, z = distr.local_grids(xbasis, zbasis)
    ex, ez = coords.unit_vector_fields(distr)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
    grad_b = d3.grad(b) + ez*lift(tau_b1) # First-order reduction

    if writeSpaceDistr:
        MPI.COMM_WORLD.Barrier()
        sleep(0.01*MPI_RANK)
        print(f"Rank {MPI_RANK}({MPI_SIZE}) :\n"
              f"\tx: {x.shape}, [{x.min(initial=np.inf)}, {x.max(initial=-np.inf)}]\n"
              f"\tz: {z.shape}, [{z.min(initial=np.inf)}, {z.max(initial=-np.inf)}]\n"
              f"\tcpu: {os.sched_getaffinity(0)}, on {socket.gethostname()}", flush=True)
        MPI.COMM_WORLD.Barrier()
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
    # problem.add_equation("b(z=0) = 2")
    # problem.add_equation("b(z=Lz) = 1")

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
                + distr.grid_layout.slices(field.domain, field.scales)
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
        infos["MPI_SIZE"] = MPI_SIZE
        if MPI_RANK == 0:
            with open(f"{dirName}/01_finalized.txt", "w") as f:
                f.write("Done !")
    except:
        log('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()

    return infos, solver

def transposeForParallel(uLoc:np.ndarray, uGlob:np.ndarray):
    uLoc = uLoc.swapaxes(0, 1).copy()
    if MPI_RANK == 0:
        uGlob = uGlob.swapaxes(0, 1).copy()
    return uLoc, uGlob


def writePySDCSolution(path:str, uLoc:np.ndarray, globShape:tuple):
    nVar, nX, nZ = globShape
    nVarLoc, nXLoc, nZLoc = uLoc.shape
    assert nVar == nVarLoc, f"rank {MPI_RANK} has inconsistent local nVar"
    assert nZ == nZLoc, f"rank {MPI_RANK} has inconsistent local nZ"

    if MPI_RANK == 0:
        uGlob = np.empty((nVar, nX, nZ), dtype=uLoc.dtype)
    else:
        uGlob = None

    msgSize = np.array(COMM_WORLD.allgather(nXLoc), dtype=int)
    msgSize *= nVar*nZ
    offsets = np.zeros(MPI_SIZE, dtype=int)
    offsets[1:] = np.cumsum(msgSize[:-1])

    if MPI_SIZE > 1:
        uLoc = uLoc.swapaxes(0, 1).copy()
        if MPI_RANK == 0:
            uGlob = uGlob.swapaxes(0, 1).copy()
        COMM_WORLD.Gatherv(
            uLoc,
            [uGlob, msgSize, offsets, MPI.DOUBLE],
            root=0)
        if MPI_RANK == 0:
            uGlob = uGlob.swapaxes(0, 1)
    else:
        uGlob = uLoc
    if MPI_RANK == 0:
        print(f" -- saving solution in {path}")
        np.save(path, uGlob)


def readPySDCSolution(path:str, uLoc:np.ndarray):
    nVar, nX, nZ = 0, 0, 0
    if MPI_RANK == 0:
        print(f" -- reading solution from {path}")
        uGlob = np.load(path)
        nVar, nX, nZ = uGlob.shape
    else:
        uGlob = None
    nVar, nX, nZ = COMM_WORLD.bcast((nVar, nX, nZ), root=0)
    nVarLoc, nXLoc, nZLoc = uLoc.shape
    assert nVar == nVarLoc, f"rank {MPI_RANK} has inconsistent local nVar"
    assert nZ == nZLoc, f"rank {MPI_RANK} has inconsistent local nZ"

    msgSize = np.array([nXLoc*nZ*nVar]*MPI_SIZE)
    offsets = np.zeros(MPI_SIZE, dtype=int)
    offsets[1:] = np.cumsum(msgSize[:-1])

    assert uLoc.shape == (nVar, nXLoc, nZ)
    if MPI_SIZE > 1:
        uLocTmp = uLoc.swapaxes(0, 1).copy()
        if MPI_RANK == 0:
            uGlob = uGlob.swapaxes(0, 1).copy()
        COMM_WORLD.Scatterv(
            [uGlob, msgSize, offsets, MPI.DOUBLE],
            uLocTmp,
            root=0)
        np.copyto(uLoc, uLocTmp.swapaxes(0, 1))
    else:
        np.copyto(uLoc, uGlob)


def runSimPySDC(dirName, Rayleigh=1e7, resFactor=1, baseDt=1e-2, seed=999,
    tBeg=0, tEnd=10, dtWrite=0.1, restartFile=None, useFNO=None,
    QI="MIN-SR-S", QE="PIC", useRK=False, timeParallel=False, nSweeps=None):

    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard
    from pySDC.implementations.sweeper_classes.imex_1st_order_MPI import imex_1st_order, imex_1st_order_MPI
    from pySDC.implementations.sweeper_classes.Runge_Kutta import ARK3
    from pySDC.implementations.problem_classes.generic_spectral import compute_residual_DAE, compute_residual_DAE_MPI
    from pySDC.helpers.fieldsIO import Rectilinear

    from cfno.simulation.sweeper import FNO_IMEX, FNO_IMEX_PINT, StepperController

    imex_1st_order.compute_residual = compute_residual_DAE
    imex_1st_order_MPI.compute_residual = compute_residual_DAE_MPI

    nNodes = 4

    Nx, Nz = 256*resFactor, 64*resFactor
    timestep = baseDt/resFactor

    nSteps = round(float(tEnd-tBeg)/timestep, ndigits=3)
    if float(tEnd-tBeg) != round(nSteps*timestep, ndigits=3):
        raise ValueError(
            f"tEnd ({tEnd}) is not divisible by timestep ({timestep}) (nSteps={nSteps})")
    nSteps = int(nSteps)
    infos = {
        "nSteps": nSteps,
        "nDOF": Nx*Nz
    }

    def log(msg):
        if MPI_RANK == 0:
            with open(f"{dirName}/simu.log", "a") as f:
                f.write(f"{dirName} -- ")
                f.write(datetime.now().strftime("%d/%m/%Y  %H:%M:%S"))
                f.write(f", MPI rank {MPI_RANK} ({MPI_SIZE})")
                f.write(f" : {msg}\n")

    os.makedirs(dirName, exist_ok=True)
    with open(f"{dirName}/00_infoSimu.txt", "w") as f:
        f.write(f"Rayleigh : {Rayleigh:1.2e}\n")
        f.write(f"seed : {seed}\n")
        f.write(f"Nx, Nz : {int(Nx)}, {int(Nz)}\n")
        f.write(f"dt : {timestep:1.2e}\n")
        f.write(f"tEnd : {tEnd}\n")
        f.write(f"dtWrite : {dtWrite}\n")

    if timeParallel:
        from pySDC.playgrounds.dedalus.sdc import initSpaceTimeMPI
        _, sComm, tComm = initSpaceTimeMPI(nProcTime=4, groupTime=False)
    else:
        tComm = None
        sComm = COMM_WORLD


    description = {
        # Step parameters
        "step_params": {
            "maxiter": 1,
        },
        # Level parameters
        "level_params": {
            "restol": -1,
            "dt": timestep,
            "nsweeps": nSweeps,
        },
        # problem parameters
        "problem_class": RayleighBenard,
        "problem_params": {
            "Rayleigh": Rayleigh,
            "nx": Nx,
            "nz": Nz,
            "dealiasing": 3/2,
            "max_cached_factorizations": nNodes+1,
            "comm": sComm,
            }
    }

    # Sweeper and its parameters
    if useRK:
        assert not timeParallel, "cannot use timeParallel with RK"
        description.update({
            "sweeper_class": ARK3,
            "sweeper_params": {},
        })
        description["level_params"]["nsweeps"] = 1
    else:
        description.update({
        "sweeper_class": imex_1st_order_MPI if timeParallel else imex_1st_order,
        "sweeper_params": {
            "quad_type": "RADAU-RIGHT",
            "num_nodes": nNodes,
            "node_type": "LEGENDRE",
            "initial_guess": "copy",
            "do_coll_update": False,
            "QI": QI,
            "QE": QE,
            "comm": tComm
            },
        })
        if useFNO is not None:
            description["sweeper_class"] = FNO_IMEX_PINT if timeParallel else FNO_IMEX
            description["sweeper_params"]["FNO"] = useFNO


    iterWrite = dtWrite/timestep
    if int(iterWrite) != round(iterWrite, ndigits=3):
        raise ValueError(
            f"dtWrite ({dtWrite}) is not divisible by timestep ({timestep}) : {iterWrite}")
    iterWrite = int(iterWrite)
    tWrite = np.linspace(tBeg, tEnd, nSteps//iterWrite+1)

    controller = StepperController(
        num_procs=1, controller_params={'logger_level': 20 if MPI_RANK == 0 else 50},
        description=description)

    prob = controller.MS[0].levels[0].prob
    sweeper = controller.MS[0].levels[0].sweep

    sX, sZ = prob.local_slice
    Rectilinear.setupMPI(
        prob.comm, [sX.start, sZ.start], [sX.stop-sX.start, sZ.stop-sZ.start])
    coordX = prob.axes[0].get_1dgrid()
    coordZ = prob.axes[1].get_1dgrid()

    if os.path.isfile(f"{dirName}/01_finalized.txt"):
        if MPI_RANK == 0:
            print(" -- simulation already finalized, skipping !")
        return infos, controller, prob

    u0 = prob.u_exact(t=0, seed=seed, noise_level=1e-3)

    uTmp = prob.itransform(u0)
    uTmp[:] = 0.0
    if restartFile is None:
        rng = np.random.default_rng(seed=seed)
        b = uTmp[2]
        z = prob.Z + 1
        b[:] = 2*rng.normal(scale=1e-3, size=b.shape)
        b *= z * (2 - z) # Damp noise at walls
        b += 2 - z # Add linear background
    else:
        file = Rectilinear.fromFile(restartFile)
        np.copyto(uTmp, file.readField(-1)[-1])
    uTmp = prob.transform(uTmp)
    np.copyto(u0, uTmp)

    u0Real = prob.itransform(u0)
    outFile = Rectilinear(u0Real.dtype, f"{dirName}/outputs.pysdc")
    outFile.setHeader(4, [coordX, coordZ])
    outFile.initialize()
    outFile.addField(0, u0Real)
    tComp = 0
    for t0, t1 in zip(tWrite[:-1], tWrite[1:]):
        tBeg = MPI.Wtime()
        u, _ = controller.run(u0=u0, t0=t0, Tend=t1, nSteps=iterWrite)
        tComp += MPI.Wtime()-tBeg
        outFile.addField(t1, prob.itransform(u))
        np.copyto(u0, u)

    infos["tComp"] = tComp
    infos["sSize"] = sComm.Get_size()
    if useFNO:
        infos["tModelEval"] = sweeper.tModelEval

    if MPI_RANK == 0:
        with open(f"{dirName}/01_finalized.txt", "w") as f:
            f.write("Done !")

    return infos, controller, prob
