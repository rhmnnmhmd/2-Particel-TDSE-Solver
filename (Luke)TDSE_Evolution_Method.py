import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
import numba
from numba import jit
from scipy.linalg import eigh_tridiagonal
import time
plt.style.use(['science', 'notebook', 'grid'])

start = time.time()

#number of x data points
Nx = 301

#x interval
dx = 1/(Nx-1)

#t interval
dt = 1e-7

#create x-domain array
x = np.linspace(0, 1, Nx)

#initial wavefunction
psi0 = np.sqrt(2)*np.sin(np.pi*x)

def V(x):
    mu, sigma = 1/2, 1/20
    return -1e4*np.exp(-(x - mu)**2 / (2*sigma**2))

d = 1/dx**2 + V(x)[1: -1]

e = -1/(2*dx**2) * np.ones(len(d) - 1)

eigenValues, eigenVectors = eigh_tridiagonal(d, e)

#get eigenenergy array
E_js = eigenValues[0: 70]

#get eigenfunction array
psi_js = np.pad(eigenVectors.T[0: 70], [(0, 0), (1, 1)], mode = "constant")

#getting the coefficient
cs = np.dot(psi_js, psi0)

#defining final output function, full wavefunction
def psi_m2(t):
    return psi_js.T@(cs*np.exp(-1j*E_js*t))

plt.plot(x, np.absolute(psi_m2(1e15*dt))**2)

plt.show()

time.sleep(1)
end = time.time()
print(f"Runtime of the program is {end - start}")