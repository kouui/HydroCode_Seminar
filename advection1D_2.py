"""
CompHydroTutorial.pdf
    - Figure 4.6
"""

import matplotlib
matplotlib.use("Qt5Agg")

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    N = 51
    x = np.linspace(0,1,N)
    dx = x[1] - x[0]
    a = np.ones(x.shape)
    a[x < 1/3] = 0.
    a[x > 2/3] = 0.
    plt.plot(x,a)

    C = 0.9; u = 1; t_max = 1
    a_origin = a.copy()
    for C in (0.5, 1.0, 10.0):
        dt = C*dx/u
        t = np.arange(0,t_max+dt,dt)
        a[:] = a_origin[:]
        Amat = np.diag(np.ones(a.size)*(1+C)) + np.diag(np.ones(a.size-1)*(-C), -1)
        Amat[0,-1] = -C
        for n,_ in enumerate(t):
            a_old = a.copy()

            #a[0] = a_old[0] - C * (a_old[0]-a_old[-1])
            #for i in range(1,N):
            #    a[i] = a_old[i] - C * (a_old[i]-a_old[i-1])

            a[:] = np.linalg.solve(Amat, a_old[:])

        plt.plot(x,a, label="C = {}".format(C))
    plt.legend()
    plt.show()
