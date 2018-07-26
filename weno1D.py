"""
this script tests the WENO scheme in 1D

CompHydroTutorial.pdf
    - Section 5.5
"""

import matplotlib
matplotlib.use("Qt5Agg")


import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

import weno_coefficients
import advection1D_3

#-----------------------------------------------------------------------------
# weno reconstruction functions
#-----------------------------------------------------------------------------

#@nb.jit(nb.float64[:](nb.int_,
#                      nb.float64[:],
#                      nb.float64[:],
#                      nb.float64[:,:],
#                      nb.float64[:,:,:]), nopython=True)
def weno_inner(order, a, C, A, sigma):

    a_weno = np.zeros(a.shape, dtype=np.double)
    beta = np.zeros((order,a.size), dtype=np.double)
    w = np.zeros(beta.shape, dtype=np.double)
    epsilon = 1E-16

    for i in range(order, a.size-order):
        a_stencils = np.zeros(order, dtype=np.double)
        alpha = np.zeros(order, dtype=np.double)

        for k in range(order):

            for l in range(order):
                for m in range(l+1):
                    beta[k,i] += sigma[k,l,m] * a[i+k-l] * a[i+k-m]

            alpha[k] = C[k] / (epsilon + beta[k, i]*beta[k, i])

            for l in range(order):
                a_stencils[k] += A[k,l] * a[i+k-l]

        w[:, i] = alpha / alpha.sum()
        a_weno[i] = np.dot(w[:,i], a_stencils[:])

    return a_weno

def weno(order, a):
    """
    Do WENO reconstruction

    Parameters
    ----------

    order : int
        The stencil width
    a : numpy array
        Scalar data to reconstruct

    Returns
    -------

    a_weno : numpy array
        Reconstructed data - boundary points are zero
    """

    C = weno_coefficients.C_all[order]
    A = weno_coefficients.A_all[order]
    sigma = weno_coefficients.sigma_all[order]

    a_weno = weno_inner(order, a, C, A, sigma)

    return a_weno

#-----------------------------------------------------------------------------
# runge kutta sub-step
#-----------------------------------------------------------------------------
def rk_substep(a, u, dx, ng):

    advection1D_3.setPeriodBCs(a[:], ng)
    f = u * a
    alpha = abs(u)
    fp = (f + alpha * a) / 2
    fm = (f - alpha * a) / 2
    fpr = np.empty(a.shape, dtype=np.double)
    fml = np.empty(a.shape, dtype=np.double)
    flux = np.empty(a.shape, dtype=np.double)
    fpr[1:] = weno(self.weno_order, fp[:-1])

#-----------------------------------------------------------------------------
# high level class
#-----------------------------------------------------------------------------
class WenoSimulation(advection1D_3.Simulation):

    def __init__(self, grid, u, C=0.8, weno_order=3):

        self.grid = grid
        #: simulation time
        # self.t = 0.0
        #: the constant advection velocity
        self.u = u
        #: CFL number
        self.C = C
        #: weno order
        self.weno_order = weno_order

    def setInitCond(self, type="tophat"):
        r""" initialize the data """
        if type == "sine_sine":
            self.grid.a[:] = np.sin( np.pi*self.grid.x -
                       np.sin(np.pi*self.grid.x) / np.pi )
        else:
            super().setInitCond(type)

    def rk_substep(self):

        g = self.grid
        advection1D_3.setPeriodBCs(g.a[:], ng)
        """
        alpha = abs(self.u)
        f = self.u * g.a

        fp   = (f + alpha * g.a) / 2
        fm   = (f - alpha * g.a) / 2
        fpr  = np.zeros(g.a.shape, dtype=np.double)
        fml  = np.zeros(g.a.shape, dtype=np.double)
        flux = np.zeros(g.a.shape, dtype=np.double)

        fpr[1:] = weno(self.weno_order, fp[:-1])
        fml[-1::-1] = weno(self.weno_order, fm[-1::-1])
        flux[1:-1] = fpr[1:-1] + fml[1:-1]
        """
        flux = np.zeros(g.a.shape, dtype=np.double)
        if self.u >= 0:
            f = self.u * g.a
            fpr = weno(self.weno_order, f[:-1])
            flux[1:-1] = fpr[0:-1]
        else:
            f = -self.u * g.a
            fml = weno(self.weno_order, f[-1::-1])
            flux[1:-1] = fml[1:-1][::-1]

        rhs = np.zeros(g.a.shape, dtype=np.double)
        rhs[1:-1] = (flux[1:-1] - flux[2:]) / g.dx
        return rhs

    def evolve(self, num_periods=1, Line=None):
        r""" evolve the linear advection equation using RK4 """
        g = self.grid

        self.t = 0.0
        tmax = num_periods * self.getPeriod()

        #: main evolution loop
        while self.t < tmax:

            #: fill the boundary conditions
            advection1D_3.setPeriodBCs(g.a[:], g.ng)

            #: get the timestep
            dt = self.getTimeStep()
            if self.t + dt > tmax:
                dt = tmax - self.t

            #: RK4, Store the data at the start of the step
            a_start = g.a.copy()
            k1 = dt * self.rk_substep()
            g.a[:] = a_start[:] + k1 / 2
            k2 = dt * self.rk_substep()
            g.a[:] = a_start[:] + k2 / 2
            k3 = dt * self.rk_substep()
            g.a[:] = a_start[:] + k3
            k4 = dt * self.rk_substep()
            g.a[:] = a_start[:] + (k1 + 2 * (k2 + k3) + k4) / 6

            if isinstance(Line,matplotlib.lines.Line2D):
                Line.set_ydata(g.a[g.ng:-g.ng])
                plt.pause(0.01)
            self.t += dt

    def evolve_scipy(self, num_periods=1, Line=None):
        r"""
        evolve the linear advection equation
        using scipy.integrate.solve_ivp
        """

if __name__ == "__main__":

    #-------------------------------------------------------------------------
    # show weno reconstruction
    #-------------------------------------------------------------------------
    if False:
        order = 3
        x = np.linspace(0,6, 61); dx = x[1]-x[0]
        y1 = np.sin(x[:])
        y1_weno = weno(order,y1[:])
        y2 = y1.copy()
        y2[30:] = y2[30] + 1.0 * x[30:] - 4
        y2_weno = weno(order,y2[:])
        fig, ax = plt.subplots(2,1, figsize=(7,7), dpi=100)
        ax[0].plot(x[order:-order],y1[order:-order], "--k")
        ax[0].plot(x[order:-order]+0.5*dx,y1_weno[order:-order], ">b", markersize=4)
        ax[1].plot(x[order:-order],y2[order:-order], "--k")
        ax[1].plot(x[order:-order]+0.5*dx,y2_weno[order:-order], ">b", markersize=4)
        plt.show()

    #-------------------------------------------------------------------------
    # compute WENO3 case
    #-------------------------------------------------------------------------

    if True:
        xmin = 0.0
        xmax = 1.0
        nx = 64
        order = 3
        ng = order+1
        g = advection1D_3.Grid1d(nx, ng, xmin=xmin, xmax=xmax)

        u = 1.0
        s = WenoSimulation(g, u, C=0.5, weno_order=3)

        fig, ax = plt.subplots(1,1, figsize=(5,5), dpi=100)
        s.setInitCond(type="tophat")
        ax.plot(g.x[g.ng:-g.ng], g.a[g.ng:-g.ng], "--k")
        line, = ax.plot(g.x[g.ng:-g.ng], g.a[g.ng:-g.ng], "-k")
        plt.pause(0.1)
        s.evolve(num_periods=5, Line=line)
        plt.show()
