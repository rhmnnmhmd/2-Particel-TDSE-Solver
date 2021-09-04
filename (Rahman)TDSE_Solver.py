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

import math

from IPython.display import HTML

#plt.style.use(["science", "notebook", "grid"])
plt.rc('savefig', dpi=300)

#POTENTIALS----------------------------------------------------------------
#soft Gaussian potentials
def potential():
    V = np.zeros((Nx, Nx))
    tmp = (dx**2)/(2*alpha**2)

    #Gaussian potential
    for m in range(1, Nx-1):
    
      for n in range(1, Nx-1):
        V[m, n] = V0*np.exp(-tmp*(m-n)*(m-n))
        #V[m, n] = V0*np.exp(-(np.abs(x1[m] - x2[n])**2) / (2*alpha**2))
    
    #hard square potential
    #for m in range(0, Nx):
      
      #for n in range(0, Nx):
        
        #if np.abs(m-n)*dx <= alpha:
          #V[m, n] = V0
            
        #else:
          #V[m, n] = 0.0
    
    return V

#INITIAL WAVEFUNCTION-----------------------------------------------------
def initialize():
    RePsi = np.zeros((Nt, Nx, Nx))
    ImPsi = np.zeros((Nt, Nx, Nx))

    X01 = x01*Nx*dx
    X02 = x02*Nx*dx

    ww = (k1*k1/(2.0*m1) + k2*k2/(2.0*m2))*0.5*dt

    for m in range(1, Nx-1):
    
      for n in range(1, Nx-1):
        y = k1*x1[m] - k2*x2[n]
        y = y - ww
        a1 = (x1[m]-X01)/sigma
        a2 = (x2[n]-X02)/sigma
        a4 = np.exp(-(a1*a1 + a2*a2) / 2)
        #A = - ((x1[m] - X01)**2 + (x2[n] - X02)**2) / (4*sigma**2)
        #B = k1*x1[m] + k2*x2[n]
        RePsi[0, m, n] = a4*np.cos(y)
        ImPsi[0, m, n] = a4*np.sin(y)
        #RePsi[0, m, n] = np.exp(A) * np.cos(B)
        #ImPsi[0, m, n] = np.exp(A) * np.sin(B)
      
    return RePsi, ImPsi

#SOLVING SE---------------------------------------------------------------
def solveSE():
    emptyArray1 = np.empty((Nt, Nx))
        
    emptyArray2 = np.empty((Nt, Nx))
    
    emptyArray3 = np.empty((Nt, Nx))
    
    Rho = np.zeros((Nx, Nx))

    for l in range(1, Nt):
    
      #real part of wavefunction
      for m in range(1, Nx-1):
        
        for n in range(1, Nx-1):
          a2 = dt*V[m, n]*ImPsi[l-1, m, n] + con2*ImPsi[l-1, m, n]
             
          a1 = dm12*(ImPsi[l-1, m+1, n] + ImPsi[l-1, m-1, n]) + dm22*(ImPsi[l-1, m, n+1] + ImPsi[l-1, m, n-1])
                   
          RePsi[l, m, n] = RePsi[l-1, m, n] - dtx*a1 + 2.0*a2

          Rho[m, n] = RePsi[l-1, m, n]*RePsi[l, m, n] + ImPsi[l-1, m, n]*ImPsi[l-1, m, n]
     
      #initialize single-particle probability
      rho1 = np.zeros(Nx) 
      rho2 = np.zeros(Nx)  
      corr = np.zeros(Nx)
    
      #sum probabilities
      SumRho = np.zeros(Nx)
        
      Ptot = 0.0
      Prel = 0.0

      p = 1

      for m in range(1, Nx-1):
        k = 1

        if p == 3:
          p = 1
          
        for n in range(1, Nx):

          if k == 3:
            k = 1
            
          Ptot = Ptot + w[k]*w[p]*Rho[m, n]

          k = k + 1
          
        p = p + 1

      for m in range(1, Nx-1):

        for n in range(1, Nx-1):
          Rho[m, n] = Rho[m, n] / Ptot

      p = 1

      #get rho1, rho2 and Rho
      for m in range(1, Nx-1):
        k = 1
            
        if p == 3:
          p = 1
    
        for n in range(1, Nx-1):
        
          if k == 3:
            k = 1
                
          #for particle 1
          rho1[m] = rho1[m] + w[k]*Rho[m, n]
               
          #for particle 2
          rho2[m] = rho2[m] + w[k]*Rho[n, m]
        
          k = k+1
        
        SumRho[m] = rho1[m] + rho2[m]
    
        if np.abs(SumRho[m]) < 1e-20:
          SumRho[m] = 0.0
    
        p = p + 1

      #correlation stuffs       
      for m in range(1, Nx-1):
        n = Nx - 1 - m
        x = m - n
        x = np.abs(x)
        
        if (rho1[m] != 0.0) and (rho2[n] != 0.0):
          corr[m] = Rho[m, n] / (rho1[m]*rho2[n])

        if (rho1[n] != 0.0) and (rho2[m] != 0.0):
          corr[m] = corr[m] + Rho[n, m]/(rho1[n]*rho2[m])

        if corr[m] != 0.0:
          corr[m] = math.log10(np.abs(corr[m]))

      emptyArray1[l-1, :] = rho1
        
      emptyArray2[l-1, :] = rho2
        
      emptyArray3[l-1, :] = SumRho
        
      #imaginary part of wave packet is next
      for m in range(1, Nx-1):
        
        for n in range(1, Nx-1):
          a2 = dt*V[m, n]*RePsi[l, m, n] + con2*RePsi[l, m, n]
                
          a1 = dm12*(RePsi[l, m+1, n]+RePsi[l, m-1, n]) + dm22*(RePsi[l, m, n+1]+RePsi[l, m, n-1])
                    
          ImPsi[l, m, n] = ImPsi[l-1, m, n] + dtx*a1 - 2.0*a2 
    
      #energy

    return emptyArray1, emptyArray2, emptyArray3

#CONSTANTS-----------------------------------------------------------------
#total space length
L = 1.401

#space step
dx = 7.0798e-3

#total space data points
Nx = L/dx
Nx = int(Nx)

#total time window
T = 0.01

#time step
dt = 1.2531e-5

#time data points
Nt = T/dt
Nt = int(Nt)

#wavenumber of 1st particle
k1 = 157.0

#wavenumber of second particle
k2 = -157.0

#width of initial wavepackets
sigma = 0.05

#initial position of particle 1
x01 = 0.25

#initial position of particle 2
x02 = 0.75

#peak potential value
V0 = 100000.0

#potential width
alpha = 0.062

#w's
w = np.zeros(3)
w[0] = dx/3
w[1] = (4/3)*dx
w[2] = (2/3)*dx

#symmetry coefficient
symmetry = 0

#mass of particle 1
m1 = 0.5

#mass of particle 2
m2 = 10*m1

dm1 = 1.0/m1
dm2 = 1.0/m2
dm12 = 0.5/m1
dm22 = 0.5/m2
dxx2 = dx**2
dtx = dt/dxx2
con = -0.5/dxx2
con2 = (dm1+dm2)*dtx

#SPACE--------------------------------------------------------------------
x1 = np.linspace(0, L, Nx)
x2 = np.linspace(0, L, Nx)

#MAIN---------------------------------------------------------------------
#potential
V = potential()

#initialize wavefunction
RePsi = initialize()
RePsi = RePsi[0]

ImPsi = initialize()
ImPsi = ImPsi[1]

#solve SE
sol = solveSE()

plot1 = sol[0]

plot2 = sol[1]

plot3 = sol[2]

#PLOTTING OR ANIMATIONS---------------------------------------------------
fig, ax = plt.subplots(1, 1)

ax.set_xlim(0, 1.5)
#ax.set_ylim(0, 0.1)

ax.plot(x1, np.abs(plot1[0, :]))
ax.plot(x1, np.abs(plot2[0, :]))

#PLOTTING OR ANIMATIONS---------------------------------------------------
# fig, ax = plt.subplots(1, 1)

# ax.set_xlim(0, 1.5)
# ax.set_ylim(0, 0.1)

# line1, = ax.plot([], [])
# line2, = ax.plot([], [])

# def init():
#     line1.set_data([], [])
#     line2.set_data([], [])

#     return line1, line2,

# def animate(i):
#     line1.set_data(x1, plot1[i, :])
#     line2.set_data(x1, plot2[i, :])

#     return line1, line2, 

# #create an animation
# anim = animation.FuncAnimation(fig, animate, init_func = init, frames = Nt, interval = 50, blit = True)

# anim.save('collission.mp4', fps=24, extra_args=['-vcodec', 'libx264'], dpi = 150)
