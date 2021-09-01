import math
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import scipy as sp
import pandas as pd
from scipy.linalg import eigh_tridiagonal
plt.style.use(['science', 'notebook', 'grid'])

#define number of points for discretization
N = 2000

#define y-interval
dy = 1/N

#define the y-domain
y = np.linspace(0, 1, N+1)

#Define a "potential"
def mL2V(y):
    return 1000*np.exp(-(y-0.5)**2)

#the matrix is a N-1 by N-1 matrix because we dont count for 
#i = 0 and i = N 
#we minus 2 from total N+1 giving us N-1

#define main diagonal
#we skip i=0 and i=N+1
d = 1/dy**2 + mL2V(y)[1: -1]

#define off-diagonal
#its length is one less than the main diagonal
e = -1/(2*dy**2) * np.ones(len(d) - 1)

#use scipy eigh_tridiagonal above
#w as eigenvalues
#v as "eigenvectors" (it returns a N-1 by N-1 matrix)
eigenValues, eigenVectors = eigh_tridiagonal(d, e)

#transpose V to get the actual eigenfunc because the actual eigenfunction 
#is across columns meaning that the 1st eigenfucn is the 1st row
eigenVectorsTranspose = eigenVectors.T

#plot the first 4 probability densitiy
fig1, ax1 = plt.subplots(2, 2)

ax1[0, 0].plot(y[1: -1], eigenVectorsTranspose[0]**2)
ax1[0, 0].set_xlabel("y = x/L")
ax1[0, 0].set_ylabel("$|\psi(y)|^2$")

ax1[0, 1].plot(y[1: -1], eigenVectorsTranspose[1]**2)
ax1[0, 1].set_xlabel("y = x/L")
ax1[0, 1].set_ylabel("$|\psi(y)|^2$")

ax1[1, 0].plot(y[1: -1], eigenVectorsTranspose[2]**2)
ax1[1, 0].set_xlabel("y = x/L")
ax1[1, 0].set_ylabel("$|\psi(y)|^2$")

ax1[1, 1].plot(y[1: -1], eigenVectorsTranspose[3]**2)
ax1[1, 1].set_xlabel("y = x/L")
ax1[1, 1].set_ylabel("$|\psi(y)|^2$")

fig1.tight_layout()

#plot eigenenergy against its corresponding energy level
fig2, ax2 = plt.subplots(1, 2)

ax2[0].bar(np.linspace(0, 10, 10), eigenValues[0: 10])
ax2[0].set_ylabel("$mL^2 E/ \hbar^2$")
ax2[0].set_xlabel("Energy Level")

ax2[1].plot(y, mL2V(y))
ax2[1].set_ylabel("$mL^2V(y)$")
ax2[1].set_xlabel("y = x/L")

fig2.tight_layout()

plt.show()