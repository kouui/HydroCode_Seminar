"""
This script defines two classes:

 -- the Grid1d class that manages a cell-centered grid and holds the
    data that lives on that grid

 -- the Simulation class that is built on a Grid1d object and defines
    everything needed to do a advection.

CompHydroTutorial.pdf
    - Section 5.2
"""
import matplotlib
matplotlib.use("Qt5Agg")
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'

import numpy as np
import numba as nb
import matplotlib.pyplot as plt


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

@nb.jit( nb.types.Tuple((nb.float64[:],nb.float64[:]))(nb.float64[:],nb.float64,nb.float64,nb.float64,nb.int64,nb.int64), nopython=True )
def getStates(a, dx, dt, u, ng, slope_type_num):
    r""" compute the left and right interface states """

    slope = np.empty(a.shape, dtype=np.double)
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

    al = np.empty(a.shape, dtype=np.double)
    ar = np.empty(a.shape, dtype=np.double)
    for i in range(ng, N-ng+1):
        #: left state on the current interface comes from zone i-1
        al[i] = a[i-1] + 0.5*dx*(1.0 - u*dt/dx)*slope[i-1]
        #: right state on the current interface comes from zone i
        ar[i] = a[i] - 0.5*dx*(1.0 + u*dt/dx)*slope[i]

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

def update(a, flux, dx, dt):
    r""" conservative update """

    a_new = np.empty(a.shape, dtype=np.double)
    N = a.shape[0]
    a_new[ng:N-ng] = a[ng:N-ng] + \
        dt/dx * (flux[ng:N-ng] - flux[ng+1:N-ng+1])

    return a_new

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

        #: main evolution loop
        while self.t < tmax:

            #: fill the boundary conditions
            setPeriodBCs(g.a[:], g.ng)

            #: get the timestep
            dt = self.getTimeStep()

            if self.t + dt > tmax:
                dt = tmax - self.t

            #: get the interface states
            al, ar = getStates(g.a[:], g.dx, dt, self.u, g.ng, slopeType2Num(self.slope_type) )

            # solve the Riemann problem at all interfaces
            flux = riemann(self.u, al[:], ar[:])

            #: do the conservative update
            a_new = update(g.a[:], flux[:], g.dx, dt)

            g.a[:] = a_new[:]

            self.t += dt

if __name__ == "__main__":

    #-------------------------------------------------------------------------
    # compare limiting and no-limiting
    #-------------------------------------------------------------------------
    xmin = 0.0
    xmax = 1.0
    nx = 64
    ng = 2

    g = Grid1d(nx, ng, xmin=xmin, xmax=xmax)

    u = 1.0
    s = Simulation(g, u, C=0.7, slope_type="centered")

    s.setInitCond(type="tophat")
    plt.plot(g.x[g.ng:-g.ng], g.a[g.ng:-g.ng], ls=":", label="exact")

    s.evolve(num_periods=5)
    plt.plot(g.x[g.ng:-g.ng], g.a[g.ng:-g.ng], label="unlimited")

    s.slope_type = "minmod"
    s.setInitCond(type="tophat")
    s.evolve(num_periods=5)
    plt.plot(g.x[g.ng:-g.ng], g.a[g.ng:-g.ng], label="minmod limiter")

    plt.legend(frameon=False, loc="best")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$a$")
    plt.title("Figure 5.4")
    plt.show()

    #-------------------------------------------------------------------------
    # convergence test
    #-------------------------------------------------------------------------
    problem = "gaussian"

    xmin = 0.0
    xmax = 1.0
    ng = 2
    N = np.array([32, 64, 128, 256, 512])

    err_god = []
    err_nolim = []
    err_lim = []
    err_lim2 = []

    u = 1.0

    for nx in N:

        #: no limiting
        g = Grid1d(nx, ng, xmin=xmin, xmax=xmax)
        s = Simulation(g, u, C=0.8, slope_type="godunov")
        s.setInitCond(type="gaussian")
        ainit = s.grid.a.copy()
        s.evolve(num_periods=5)
        err_god.append(g.norm(g.a - ainit))

        #: no limiting
        s.slope_type = "centered"
        s.setInitCond(type="gaussian")
        s.evolve(num_periods=5)
        err_nolim.append(g.norm(g.a - ainit))

        #: MC limiting
        s.slope_type = "MC"
        s.setInitCond(type="gaussian")
        s.evolve(num_periods=5)
        err_lim.append(g.norm(g.a - ainit))

        #: minmod limiting
        s.slope_type = "minmod"
        s.setInitCond(type="gaussian")
        s.evolve(num_periods=5)
        err_lim2.append(g.norm(g.a - ainit))

        print(g.dx, nx, err_nolim[-1], err_lim[-1], err_lim2[-1])

    plt.scatter(N, err_god, label="Godunov", color="C0")
    plt.scatter(N, err_nolim, label="unlimited center", color="C1")
    plt.scatter(N, err_lim, label="MC", color="C2")
    plt.scatter(N, err_lim2, label="minmod", color="C3")
    plt.plot(N, err_god[len(N)-1]*(N[len(N)-1]/N),
             color="k", label=r"$\mathcal{O}(\Delta x)$")
    plt.plot(N, err_nolim[len(N)-1]*(N[len(N)-1]/N)**2,
             color="0.5", label=r"$\mathcal{O}(\Delta x^2)$")

    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlabel("N")
    plt.ylabel(r"$\| a^\mathrm{final} - a^\mathrm{init} \|_2$",
               fontsize=16)
    plt.legend(frameon=False, loc="best", fontsize="small")
    plt.show()

    #-------------------------------------------------------------------------
    # different limiters: run both the Gaussian and tophat
    #-------------------------------------------------------------------------
    xmin = 0.0
    xmax = 1.0
    nx = 32
    ng = 2

    u = 1.0

    g= Grid1d(nx, ng, xmin=xmin, xmax=xmax)

    for p in ["gaussian", "tophat"]:

        #: godunov
        s = Simulation(g, u, C=0.8, slope_type="godunov")
        s.setInitCond(type=p)
        ainit = s.grid.a.copy()
        s.evolve(num_periods=5)

        plt.subplot(231)
        plt.plot(g.x[g.ng:-g.ng], ainit[g.ng:-g.ng], ls=":")
        plt.plot(g.x[g.ng:-g.ng], g.a[g.ng:-g.ng])
        plt.title("piecewise constant")

        #: centered
        s.slope_type = "centered"
        s.setInitCond(type=p)
        s.evolve(num_periods=5)

        plt.subplot(232)
        plt.plot(g.x[g.ng:-g.ng], ainit[g.ng:-g.ng], ls=":")
        plt.plot(g.x[g.ng:-g.ng], g.a[g.ng:-g.ng])
        plt.title("centered (unlimited)")

        #: minmod
        s.slope_type = "minmod"
        s.setInitCond(type=p)
        s.evolve(num_periods=5)

        plt.subplot(233)
        plt.plot(g.x[g.ng:-g.ng], ainit[g.ng:-g.ng], ls=":")
        plt.plot(g.x[g.ng:-g.ng], g.a[g.ng:-g.ng])
        plt.title("minmod limiter")

        #: MC
        s.slope_type = "MC"
        s.setInitCond(type=p)
        s.evolve(num_periods=5)

        plt.subplot(234)
        plt.plot(g.x[g.ng:-g.ng], ainit[g.ng:-g.ng], ls=":")
        plt.plot(g.x[g.ng:-g.ng], g.a[g.ng:-g.ng])
        plt.title("MC limiter")

        #: superbee
        s.slope_type = "MC"
        s.setInitCond(type=p)
        s.evolve(num_periods=5)

        plt.subplot(235)
        plt.plot(g.x[g.ng:-g.ng], ainit[g.ng:-g.ng], ls=":")
        plt.plot(g.x[g.ng:-g.ng], g.a[g.ng:-g.ng])
        plt.title("superbee limiter")

        f = plt.gcf()
        f.set_size_inches(10.0,7.0)
        plt.tight_layout()
        plt.show()
