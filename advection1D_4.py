"""
This script update time steps by integrating ODE

CompHydroTutorial.pdf
    - Section 5.3
"""
import matplotlib
matplotlib.use("Qt5Agg")
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'

import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp


#-----------------------------------------------------------------------------
# flux limiter
#-----------------------------------------------------------------------------
@nb.jit(nb.float64(nb.float64,nb.float64), nopython=True)
def minmod(a, b):

    if abs(a) < abs(b) and a*b > 0.0:
        return a
    elif abs(b) < abs(a) and a*b > 0.0:
        return b
    else:
        return 0.0

@nb.jit(nb.float64(nb.float64,nb.float64), nopython=True)
def maxmod(a, b):

    if abs(a) > abs(b) and a*b > 0.0:
        return a
    elif abs(b) > abs(a) and a*b > 0.0:
        return b
    else:
        return 0.0
#-----------------------------------------------------------------------------
# Simulation function
#-----------------------------------------------------------------------------
def setPeriodBCs(a, ng):
    r""" set period boundary condition for array a """

    a[:ng] = a[-2*ng:-ng]
    a[-ng:] = a[ng:2*ng]

def slopeType2Num(slope_type):

    if slope_type == "godunov":
        return 0

    elif slope_type == "centered":
        return 1

    elif slope_type == "minmod":
        return 2

    elif slope_type == "MC":
        return 3

    elif slope_type == "superbee":
        return 4

@nb.jit( nb.types.Tuple((nb.float64[:],nb.float64[:]))(nb.float64[:],nb.float64,nb.int64,nb.int64), nopython=True )
def getStates(a, dx, ng, slope_type_num):
    r""" compute the left and right interface states """

    slope = np.zeros(a.shape, dtype=np.double)
    N = a.shape[0]

    #if slope_type == "godunov":
    if slope_type_num == 0:

        #: piecewise constant = 0 slopes
        slope[:] = 0.0

    #elif slope_type == "centered":
    elif slope_type_num == 1:

        #: unlimited centered difference slopes
        for i in range(ng-1, N-ng+1):
            slope[i] = 0.5*(a[i+1] - a[i-1])/dx

    #elif slope_type == "minmod":
    elif slope_type_num == 2:

        #: minmod limited slope
        for i in range(ng-1, N-ng+1):
            slope[i] = minmod( (a[i] - a[i-1])/dx,(a[i+1] - a[i])/dx )

    #elif slope_type == "MC":
    elif slope_type_num == 3:

        #: MC limiter
        for i in range(ng-1, N-ng+1):
            slope[i] = minmod(minmod( 2.0*(a[i] - a[i-1])/dx,2.0*(a[i+1] - a[i])/dx ),0.5*(a[i+1] - a[i-1])/dx)

    #elif slope_type == "superbee":
    elif slope_type_num == 4:

        #: superbee limiter
        for i in range(ng-1, N-ng+1):
            A = minmod( (a[i+1] - a[i])/dx,2.0*(a[i] - a[i-1])/dx )

            B = minmod( (a[i] - a[i-1])/dx,2.0*(a[i+1] - a[i])/dx )

            slope[i] = maxmod(A, B)

    al = a[:] - 0.5*slope[:]*dx
    ar = a[:] + 0.5*slope[:]*dx
    #al = np.empty(a.shape, dtype=np.double)
    #ar = np.empty(a.shape, dtype=np.double)
    #for i in range(ng, N-ng+1):
    #    al[i] = a[i] - 0.5*slope[i]*dx
    #    ar[i] = a[i] + 0.5*slope[i]*dx

    return al, ar

def riemann(u, al, ar):
    r"""
    Riemann problem for advection -- this is simply upwinding,
    but we return the flux
    """
    if u > 0:
        return u * al
    else:
        return u * ar

def integrand(a, t, dx, u, ng, slope_type_num):

    setPeriodBCs(a[:], ng)
    al, ar = getStates(a[:], dx, ng, slope_type_num )
    flux = riemann(u, ar[ng-1:-ng], al[ng:-ng+1])

    dadt = np.zeros(a.shape, dtype=np.double)
    dadt[ng:-ng] = - (flux[1:] - flux[:-1]) / dx
    return dadt


#-----------------------------------------------------------------------------
# Grid class
#-----------------------------------------------------------------------------
class Grid1d:

    def __init__(self, nx, ng, xmin=0.0, xmax=1.0):

        self.ng = ng
        self.nx = nx

        self.xmin = xmin
        self.xmax = xmax

        #: physical coords -- cell-centered, left and right edges
        dx = (xmax - xmin)/(nx); self.dx = dx
        N = nx + 2*ng; self.N = N
        self.xl = np.linspace(xmin-ng*dx, xmax+ng*dx, N)
        self.x = self.xl + 0.5*dx
        self.xr = self.xl + dx

        # storage for the solution
        self.a = np.empty((N), dtype=np.float64)

    def norm(self, e):
        r""" return the norm of quantity e which lives on the grid """
        if len(e) != 2*self.ng + self.nx:
            return None

        #return np.sqrt(self.dx*np.sum(e[self.ilo:self.ihi+1]**2))
        return np.max(abs(e[self.ng:-self.ng]))

#-----------------------------------------------------------------------------
# Simulation class
#-----------------------------------------------------------------------------
class Simulation:

    def __init__(self, grid, u, C=0.8, slope_type="centered"):

        self.grid = grid
        #: simulation time
        self.t = 0.0
        #: the constant advection velocity
        self.u = u
        #: CFL number
        self.C = C
        #: flux constructor type
        self.slope_type = slope_type

    def setInitCond(self, type="tophat"):
        r""" initialize the data """
        if type == "tophat":
            self.grid.a[:] = 0.0
            self.grid.a[ np.logical_and(self.grid.x >= 0.333,self.grid.x <= 0.666) ] = 1.0
        elif type == "sine":
            self.grid.a[:] = np.sin(2.0*np.pi*self.grid.x/(self.grid.xmax-self.grid.xmin))
        elif type == "gaussian":
            al = 1.0 + np.exp(-60.0*(self.grid.xl - 0.5)**2)
            ar = 1.0 + np.exp(-60.0*(self.grid.xr - 0.5)**2)
            ac = 1.0 + np.exp(-60.0*(self.grid.x - 0.5)**2)

            self.grid.a[:] = (1./6.)*(al + 4*ac + ar)

    def getTimeStep(self):
        r""" return the advective timestep """
        return self.C*self.grid.dx/self.u

    def getPeriod(self):
        r""" return the period for advection with velocity u """
        return (self.grid.xmax - self.grid.xmin)/self.u

    #@profile
    def evolve(self, num_periods=1):
        r""" evolve the linear advection equation """

        g = self.grid

        self.t = 0.0
        tmax = num_periods * self.getPeriod()

        t = (self.t, tmax)
        sol = solve_ivp(fun=lambda t, a: integrand(a, t, g.dx, self.u, g.ng, slopeType2Num(self.slope_type)),
                        t_span=t, y0=g.a[:], method="RK45")

        return sol

if __name__ == "__main__":

    xmin = 0.0
    xmax = 1.0
    nx = 64
    ng = 2

    g = Grid1d(nx, ng, xmin=xmin, xmax=xmax)

    u = 1.0
    s = Simulation(g, u, C=0.7, slope_type="godunov")

    s.setInitCond(type="tophat")
    plt.plot(g.x[g.ng:-g.ng], g.a[g.ng:-g.ng], ls=":", label="exact")

    for slope_type in ("godunov", "centered", "minmod", "MC", "superbee"):

        s.slope_type = slope_type
        sol = s.evolve(num_periods=2)
        plt.plot(g.x[g.ng:-g.ng], sol.y[g.ng:-g.ng, -1], label=slope_type)



    plt.legend(frameon=False, loc="best")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$a$")
    plt.title("Figure 5.4")
    plt.show()
