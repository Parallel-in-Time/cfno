import h5py
import random
import numpy as np
from datetime import datetime
import dedalus.public as d3


def runSim(dirName, Rayleigh, resFactor, baseDt=1e-2/2, seed=999,
            tEnd=150, writeVort=False):
    """
    Run RBC simulation in a given folder.

    Parameters
    ----------
    dirName : str
        Name of directory where to store snapshots and run infos
    Rayleigh : float
        Rayleigh number.
    resFactor : int
        Resolution factor, considering a base space grid size of (256,64).
    baseDt : float, optional
        Base time-step for the base space resolution. The default is 1e-2/2.
    seed : int, optional
        Seed for the random noise in the initial solution. The default is 999.
    """
    def log(msg):
        with open(f"{dirName}/simu.log", "a") as f:
            f.write(f"{dirName} -- ")
            f.write(datetime.now().strftime("%d/%m/%Y  %H:%M:%S"))
            f.write(f" : {msg}\n")

    # Parameters
    Lx, Lz = 4, 1
    Nx, Nz = 256*resFactor, 64*resFactor
    timestep = baseDt/resFactor

    Prandtl = 1
    dealias = 3/2
    stop_sim_time = tEnd
    timestepper = d3.RK443
    dtype = np.float64

    with open(f"{dirName}/00_infoSimu.txt", "w") as f:
        f.write(f"Rayleigh : {Rayleigh:1.2e}\n")
        f.write(f"Seed : {seed}\n")
        f.write(f"Nx, Nz : {int(Nx)}, {int(Nz)}\n")
        f.write(f"dt : {timestep:1.2e}\n")
        f.write(f"tEnd : {tEnd}\n")

    # Bases
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=dtype)
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
    solver.stop_sim_time = stop_sim_time

    # Initial conditions
    b.fill_random('g', seed=seed, distribution='normal', scale=1e-3) # Random noise
    b['g'] *= z * (Lz - z) # Damp noise at walls
    b['g'] += Lz - z # Add linear background

    # Analysis
    snapshots = solver.evaluator.add_file_handler(
        dirName, sim_dt=0.1, max_writes=1600)
    snapshots.add_task(u, name='velocity')
    snapshots.add_task(b, name='buoyancy')
    snapshots.add_task(p, name='pressure')
    if writeVort:
        snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

    # Main loop
    try:
        log('Starting main loop')
        while solver.proceed:
            solver.step(timestep)
            if (solver.iteration-1) % 100 == 0:
                log(f'Iteration={solver.iteration}, Time={solver.sim_time}, dt={timestep}')
    except:
        log('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()




