#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMEX Sweeper allowing step initialization with a NN model
"""
import numpy as np
import itertools

from cfno.training.pySDC import FourierNeuralOp

from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.core.errors import ControllerError


class StepperController(controller_nonMPI):

    def run(self, u0, t0, Tend, nSteps):
        """
        Main driver for running the serial version of SDC, MSSDC, MLSDC and PFASST (virtual parallelism)

        Args:
           u0: initial values
           t0: starting time
           Tend: ending time

        Returns:
            end values on the finest level
            stats object containing statistics for each step, each level and each iteration
        """

        # some initializations and reset of statistics
        uend = None
        num_procs = len(self.MS)
        for hook in self.hooks:
            hook.reset_stats()

        # initial ordering of the steps: 0,1,...,Np-1
        slots = list(range(num_procs))

        # initialize time variables of each step
        time = [t0 + sum(self.MS[j].dt for j in range(p)) for p in slots]

        # determine which steps are still active : all
        active = [True for p in slots]

        if not any(active):
            raise ControllerError('Nothing to do, check t0, dt and Tend.')

        # compress slots according to active steps, i.e. remove all steps which have times above Tend
        active_slots = list(itertools.compress(slots, active))

        # initialize block of steps with u0
        self.restart_block(active_slots, time, u0)

        for hook in self.hooks:
            hook.post_setup(step=None, level_number=None)

        # call pre-run hook
        for S in self.MS:
            for hook in self.hooks:
                hook.pre_run(step=S, level_number=0)

        # main loop: compute nSteps time-steps ...
        for _ in range(nSteps):
            MS_active = [self.MS[p] for p in active_slots]
            done = False
            while not done:
                done = self.pfasst(MS_active)

            restarts = [S.status.restart for S in MS_active]
            restart_at = np.where(restarts)[0][0] if True in restarts else len(MS_active)
            if True in restarts:  # restart part of the block
                # initial condition to next block is initial condition of step that needs restarting
                uend = self.MS[restart_at].levels[0].u[0]
                time[active_slots[0]] = time[restart_at]
                self.logger.info(f'Starting next block with initial conditions from step {restart_at}')

            else:  # move on to next block
                # initial condition for next block is last solution of current block
                uend = self.MS[active_slots[-1]].levels[0].uend
                time[active_slots[0]] = time[active_slots[-1]] + self.MS[active_slots[-1]].dt

            for S in MS_active[:restart_at]:
                for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
                    C.post_step_processing(self, S, MS=MS_active)

            for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
                [C.prepare_next_block(self, S, len(active_slots), time, Tend, MS=MS_active) for S in self.MS]

            # setup the times of the steps for the next block
            for i in range(1, len(active_slots)):
                time[active_slots[i]] = time[active_slots[i] - 1] + self.MS[active_slots[i] - 1].dt

            # determine new set of active steps and compress slots accordingly
            active = [True for p in slots]
            active_slots = list(itertools.compress(slots, active))

            # restart active steps (reset all values and pass uend to u0)
            self.restart_block(active_slots, time, uend)

        # call post-run hook
        for S in self.MS:
            for hook in self.hooks:
                hook.post_run(step=S, level_number=0)

        for S in self.MS:
            for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
                C.post_run_processing(self, S, MS=MS_active)

        return uend, self.return_stats()


class FNO_IMEX(imex_1st_order):

    def __init__(self, params):
        super().__init__(params)

        assert "FNO" in params, "need FNO parameters in sweeper params"
        self.model = FourierNeuralOp(**params["FNO"])

        M = params["num_nodes"]
        self.uPrev = [None for _ in range(M)]


    @property
    def nNodes(self):
        return len(self.uPrev)

    def predict(self):
        # get current level and problem description
        L = self.level
        P = L.prob

        # evaluate RHS at left point
        L.f[0] = P.eval_f(L.u[0], L.time)

        for m in range(1, self.nNodes + 1):
            if self.uPrev[0] is None:
                # First step, no previous node values stored
                L.u[m] = P.dtype_u(L.u[0])
                L.f[m] = P.dtype_f(L.f[0])
            else:
                # Use FNO update
                L.u[m] = P.u_init
                uTmp = P.transform(self.model(P.itransform(self.uPrev[m-1])))
                uTmp *= 2 # pySDC has a x2 scaling in space
                P.xp.copyto(L.u[m], uTmp)
                L.u[m][:3] = P.solve_system(L.u[m], dt=1e-10, u0=None)[:3]
                L.f[m] = P.eval_f(L.u[m], L.time + L.dt * self.coll.nodes[m - 1])

        # indicate that this level is now ready for sweeps
        L.status.unlocked = True
        L.status.updated = True


    def update_nodes(self):
        super().update_nodes()

        # Store the node values into uPrev for next sweep
        for m in range(self.nNodes):
            self.uPrev[m] = self.level.prob.dtype_u(self.level.u[m+1])
