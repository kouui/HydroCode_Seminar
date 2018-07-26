"""
This script tests 2D advection: Dimensionally split

CompHydroTutorial.pdf
    - Figure 5.14
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
class Grid2d:

    def __init__(self, nx, ny, ng, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0):

        self.ng = ng
        self.nx = nx
        self.ny = ny

        self.xmin = xmin
        self.xmax = xmax

        self.ymin = ymin
        self.ymax = ymax

        #: physical coords -- cell-centered, left and right edges
        #: x direction
        dx = (xmax - xmin)/(nx); self.dx = dx
        Nx = nx + 2*ng; self.Nx = Nx
        self.xl = np.linspace(xmin-ng*dx, xmax+ng*dx, Nx)
        self.x = self.xl + 0.5*dx
        self.xr = self.xl + dx
        #: y direction
        dy = (ymax - ymin)/(ny); self.dy = dy
        Ny = ny + 2*ng; self.Ny = Ny
        self.yl = np.linspace(ymin-ng*dy, ymax+ng*dy, Ny)
        self.y = self.yl + 0.5*dy
        self.yr = self.yl + dy


        # storage for the solution
        self.a = np.empty((Ny,Nx), dtype=np.float64)



#-----------------------------------------------------------------------------
# Simulation class
#-----------------------------------------------------------------------------
class Simulation:

    def __init__(self, grid, ux, uy, C=0.8, slope_type="centered"):

        self.grid = grid
        #: simulation time
        self.t = 0.0
        #: the constant advection velocity
        self.ux = ux
        self.uy = uy
        #: CFL number
        self.C = C
        #: flux constructor type
        self.slope_type = slope_type

    def setInitCond(self, type="tophat"):
        r""" initialize the data """
        nx, ny, ng = self.grid.nx, self.grid.ny, self.grid.ng
        x1, x2 = nx//3 + ng, 2*nx//3 + ng
        y1, y2 = ny//3 + ng, 2*ny//3 + ng
        if type == "tophat":
            self.grid.a[:,:] = 0.0
            self.grid.a[y1:y2,x1:x2] = 1.0

    def getTimeStep(self, ds, u):
        r""" return the advective timestep """
        return self.C*ds/u

    def getPeriod(self, smax, smin, u):
        r""" return the period for advection with velocity u """
        return (smax - smin)/u

    def evolve(self, num_periods=1, im=None):
        r""" evolve the linear advection equation """

        g = self.grid

        self.t = 0.0
        dtx = self.getTimeStep(g.dx,self.ux)
        dty = self.getTimeStep(g.dy,self.uy)
        dt = min(dtx,dty)
        Tx = self.getPeriod(g.xmax, g.xmin, self.ux)
        Ty = self.getPeriod(g.ymax, g.ymin, self.uy)
        Period = max(Tx, Ty)
        tmax = num_periods * Period

        #: main evolution loop
        while self.t < tmax:

            #-----------------------------------------------------------------
            # x direction
            #-----------------------------------------------------------------
            ds, u = g.dx, self.ux
            for i in range(g.ng, g.ng+g.nx):
                a = g.a[i,:]
                #: fill the boundary conditions
                setPeriodBCs(a[:], g.ng)

                #: get the interface states
                al, ar = getStates(a[:], ds, dt, u, g.ng, slopeType2Num(self.slope_type) )

                # solve the Riemann problem at all interfaces
                flux = riemann(u, al[:], ar[:])

                #: do the conservative update
                a_new = update(a[:], flux[:], ds, dt)

                a[:] = a_new[:]

            #-----------------------------------------------------------------
            # y direction
            #-----------------------------------------------------------------
            ds, u = g.dy, self.uy
            for i in range(g.ng, g.ng+g.ny):
                a = g.a[:,i]
                #: fill the boundary conditions
                setPeriodBCs(a[:], g.ng)

                #: get the interface states
                al, ar = getStates(a[:], ds, dt, u, g.ng, slopeType2Num(self.slope_type) )

                # solve the Riemann problem at all interfaces
                flux = riemann(u, al[:], ar[:])

                #: do the conservative update
                a_new = update(a[:], flux[:], ds, dt)

                a[:] = a_new[:]

            if isinstance(im,matplotlib.image.AxesImage):
                im.set_data(g.a[g.ng:-g.ng, g.ng:-g.ng])
                plt.pause(0.01)

            self.t += dt

if __name__ == "__main__":

    #-------------------------------------------------------------------------
    # compare limiting and no-limiting
    #-------------------------------------------------------------------------
    nx = 32
    ny = 32
    ng = 2
    g = Grid2d(nx, ny, ng)

    ux, uy = 1.0, 0.5
    s = Simulation(g, ux, uy, C=0.8, slope_type="centered")

    fig, ax = plt.subplots(1,1, figsize=(5,5), dpi=100)

    s.setInitCond(type="tophat")
    im = ax.imshow(g.a[g.ng:-g.ng, g.ng:-g.ng], cmap="gray", origin="lower")
    plt.pause(0.1)

    s.evolve(num_periods=2, im=im)

    #plt.show()
