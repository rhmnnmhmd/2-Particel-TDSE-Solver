import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
import numba
from numba import jit
from scipy.linalg import eigh_tridiagonal
import time
plt.style.use(["science", "notebook", "grid"])
from matplotlib.animation import FuncAnimation

start = time.time()

#CONSTANTS
#--------------------------------------------------------------------------
#number of x data points
Nx = 301

#number of t data points
Nt = 100000

#dt/(dx)^2 must be really small for this finite difference method to work
#x interval
dx = 1/(Nx-1)

#t interval
dt = 1e-7

#displacement of potential from origin
mu = 1/2

#"width" of potential
sigma = 1/20

#INITIAL ARRAYS
#--------------------------------------------------------------------------
#create x-domain array
x = np.linspace(0, 1, Nx)

#initial wavefunction psi(x, 0)
psi0 = np.sqrt(2)*np.sin(np.pi*x)

#(Gaussian) potential V(x)
V = -1e4*np.exp(-(x - mu)**2 / (2*sigma**2))

#create 2-D array of zeros for psi(x, t)
psi = np.zeros((Nt, Nx))

#psi(x, t=0)
psi[0] = psi0

@numba.jit("c16[:, :](c16[:, :])", nopython = True, nogil = True)
def compute_psi(psi):
    for t in range(0, Nt-1):
        
        for i in range(1, Nx-1):
            psi[t+1][i] = psi[t][i] + 1j/2 * dt/dx**2 * (psi[t][i+1] - 2*psi[t][i] + psi[t][i-1]) - 1j*dt*V[i]*psi[t][i]
        
        normal = np.sum(np.absolute(psi[t+1])**2)*dx
        
        for i in range(1, Nx-1):
            psi[t+1][i] = psi[t+1][i]/normal
            
    return psi

#getting the output eigenfunction
psi_m1 = compute_psi(psi.astype(complex))

#PLOTTING/ANIMATION------------------------------------------------------------------
fig, ax = plt.subplots(1, 1)

ax.set_xlim(0, 1)
ax.set_ylim(0, 2)

line, = ax.plot([], [])

def init():
    line.set_data([], [])
    return line,

def animate(i):
    probDensity = np.absolute(psi_m1[i, 0: ])**2
    line.set_data(x, probDensity)
    
    return line,

anim = FuncAnimation(fig, animate, init_func=init, frames=Nt, interval=1/30, blit=True)

anim.save('probDensity.gif', writer='imagemagick')

time.sleep(1)
end = time.time()
print(f"Runtime of the program is {end - start}")