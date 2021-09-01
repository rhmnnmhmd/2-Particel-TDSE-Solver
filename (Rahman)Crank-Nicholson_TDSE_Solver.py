#IMPORT LIBRARY PACKAGES---------------------------------------------------
import sys

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation

import numba
from numba import jit, njit

import time

import scipy as sp
from scipy import integrate
from scipy import sparse
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import spdiags
from scipy.sparse.linalg import splu

import math

from IPython.display import HTML

#start time---------------------------------------------------------------
start = time.time()

#functions-----------------------------------------------------------------
#potential
def potential():
    #hard square potential
    V = np.zeros(J)
    
    for m in range(0, J):
            
        if np.abs(m-n)*dx <= alpha:
            V[m] = -100
            
        else:
            V[m] = 0.0
            
    return V
    
#space and time discretization---------------------------------------------
#number of space lattice
J = 1024

#length of spatial dimensiom
L = 20

#space increment
dx = L/J

#time increment
dt = 0.0001

#length of temporal dimension
T = 1.6

#number of time lattice
Nt = T/dt
Nt = int(Nt)

#constants----------------------------------------------------------------
sigma0 = 1.0

k0 = 7.0

xi0 = 0

#initial obejcts----------------------------------------------------------
#spatial domain
x = np.linspace(-L/2, L/2, J)

#potential of array of zeros
V = 0.02*x**2

#array of ones
o = np.ones((J), complex)

#alpha
alpha = (1j*dt) / (2 * dx**2) * o

#diagonals position
diags = np.array([-1, 0, +1])


##creating U1
#xi
xi = o + 1j*dt/2*(2/(dx**2)*o + V)

#diagonal entries
vecs1 = np.array([-alpha, xi, -alpha])

#U1
U1 = spdiags(vecs1, diags, J, J)

#convert to different sparse format
U1 = U1.tocsc()


##creating U2
#gamma
gamma = o - 1j*dt/2*(2/(dx**2)*o + V)

#diagonal entries
vecs2 = np.array([alpha, gamma, alpha])

#U2
U2 = spdiags(vecs2, diags, J, J)


#compute LU decomposition
LU = splu(U1)

#intitial wavefunc at t = 0
psi0 = (1/(sigma0**2*np.pi))**(1/4) * np.exp(1j*k0*x - (x - xi0)**2/(2*sigma0**2))

#general wavefucntion at any time t
psi = np.zeros((Nt, J), complex)

#putting initial wavefunc in the 2D general wavefunc array
psi[0, :] = psi0

#loop for each time step
for n in range(0, Nt-1):
    b = U2.dot(psi[n, :])
    psi[n+1, :] = LU.solve(b)
    
#plotting-----------------------------------------------------------------
fig, ax = plt.subplots(1, 1)

ax.plot(x, V)
ax.plot(x, np.abs(psi[15999, :])**2)

#end time-----------------------------------------------------------------
time.sleep(1)
end = time.time()
print(f"Runtime is {end-start} seconds")