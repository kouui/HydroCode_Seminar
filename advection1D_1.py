"""
CompHydroTutorial.pdf
    - Figure 4.3
    - Figure 4.4
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
    #plt.plot(x,a)
    #plt.show()

    C = 0.9; u = 1; t_max = 1

    a_origin = a.copy()
    for C in (1, 0.9, 0.5, 0.1):
        dt = C*dx/u
        t = np.arange(0,t_max+dt,dt)
        a[:] = a_origin[:]
        for n,_ in enumerate(t):
            #plt.clf()
            a_old = a.copy()
            a[0] = a_old[0] - C * (a_old[0]-a_old[-1])
            for i in range(1,N):
                a[i] = a_old[i] - C * (a_old[i]-a_old[i-1])
            #plt.plot(x,a)
            #plt.pause(0.05)
        plt.plot(x,a, label="C = {}".format(C))
    plt.legend()
    plt.show()
