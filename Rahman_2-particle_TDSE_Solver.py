import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
import numba
from numba import jit
from scipy.linalg import eigh_tridiagonal
import time
import math
plt.style.use(['science', 'notebook', 'grid'])

start = time.time()
#-------------------------------------------------------------------------
#define functions#
#--------------------------------------------------------------------------
#createwavefunction psi
def initialize():
    x1 = 0.0
    X01 = x01*xDim*dx
    X02 = x02*xDim*dx
    
    RePsi = np.zeros(shape = (xDim, xDim, 2))
    ImPsi = np.zeros(shape = (xDim, xDim, 2))
    
    for i in range(1, N1+1):
        x1 = x1 + dx
        x2 = 0.0
        
        for j in range(1, N2+1):
            x2 = x2+dx
    		y = k1*x1-k2*x2
    		y = y-ww
    		a1 = (x1-X01)/sig
    		a2 = (x2-X02)/sig
    		a4 = np.exp(-(a1*a1+a2*a2)/2)
    		RePsi[i, j, 0] = a4*np.cos(y)
    		ImPsi[i, j, 0] = a4*np.sin(y)
    
    #Set wavefunction to zero on boundary (should never be called anyway)
    for j in range(0, N1+2):
        RePsi[N1+1][j][0] = 0.0
    	RePsi[0][j][0] = 0.0
        
    for i in range(1, N2+1):
        RePsi[i, N2+1, 0] = 0.0
        RePsi[i, 0, 0] = 0.0
        
    #Find the initial (unnormalized) energy at roughly t = 0
    Eri = 0.0
    Eii = 0.0
    p = 1       
    
    for i in range(1, N1+1):
        k = 1
        
        if p == 3:
            p = 1
            
        for j in range(1, N2+1):
            
            if k == 3:
                k = 1
            
            if alpha >=  np.abs(i-j)*dx:
                v = vmax()
                
            else:
                v = 0.0
                
            a1 = -2*(dm1+dm2+dx*dx*v)*(RePsi[i,j,0]*RePsi[i,j,0]+ImPsi[i,j,0]*ImPsi[i,j,0])
            
    		a2 = dm1*(RePsi[i,j,0]*(RePsi[i+1,j,0]+RePsi[i-1,j,0])+ImPsi[i,j,0]*(ImPsi[i+1,j,0]+ImPsi[i-1,j,0]))
            
    		a3 = dm2*(RePsi[i,j,0]*(RePsi[i,j+1,0]+RePsi[i,j-1,0])+ImPsi[i,j,0]*(ImPsi[i,j+1,0]+ImPsi[i,j-1,0]))
            
    		ai1 = dm1*(RePsi[i,j,0]*(ImPsi[i+1,j,0]+ImPsi[i-1,j,0])-ImPsi[i,j,0]*(RePsi[i+1,j,0]+RePsi[i-1,j,0]))
            
    		ai2 = dm2*(RePsi[i,j,0]*(ImPsi[i,j+1,0]+ImPsi[i,j-1,0])-ImPsi[i,j,0]*(RePsi[i,j+1,0]+RePsi[i,j-1,0]))
            
    		Eri = Eri + w[k]*w[p]*con*(a1 + a2 + a3)
            
    		Eii = Eii + w[k]*w[p]*con*(ai1 + ai2)
            
    		k = k+1
            
        p = p + 1
        
    return Eri, Eii, RePsi, ImPsi

#probability
def probability():
    rho_1 = np.zeros(xDim)
    rho_2 = np.zeros(xDim)
    
    #initialize the single probability arrays to zero
    for i in range(0, N1+2):
        rho_1[i] = 0.0
        rho_2[i] = 0.0
        
    #Normalize Rho, important for Correlation functions
    Ptot = 0.0
    Prel = 0.0 
    p = 1
    
    for i in range(1, N1+1):
        k = 1
        
        if p == 3:
            p = 1 
        
        for j in range(1, N2+1):
            Rho = RePsi[i, j, 0]*RePsi[i, j, 1] + ImPsi[i, j, 0]*ImPsi[i, j, 0]
            
            #impose symmetry/antisymmetry
            Rho = Rho + symmetry*(RePsi[i,j,0]*RePsi[j,i,1]+ ImPsi[i,j,0]*ImPsi[j,i,0])
            
            if k == 3:
                k = 1
            
            Ptot = Ptot + w[k]*w[p]*Rho;
            
    		k = k + 1
            
        p = p + 1 
        
    if n == 1:
        Ptot_i = Ptot
        
    #renormalize Rho
    for i in range(1, N1+1):
        
        for j in range(1, N2+1):
            Rho[i, j] = Rho[i, j]/Ptot
    
    #Integrate out 1D probabilites from 2D 
    p = 1
    
    for i in range(1, N1+1):
        k = 1 
        
        if p == 3:
            p = 1
        
        for j in range(1, N2+1):
            
            if k == 3:
                k = 1
            
            Rho = RePsi[i][j][0]*RePsi[i][j][1]+ImPsi[i][j][0]*ImPsi[i][j][0];
            
            #Impose symmetry or antisymmetry
    		Rho = Rho + symmetry*(RePsi[i][j][0]*RePsi[j][i][1]+ImPsi[i][j][0]*ImPsi[j][i][0]);
    		
            rho_1[i] = rho_1[i] + w[k]*Rho;
    		
            Rho = RePsi[j][i][0]*RePsi[j][i][1]+ImPsi[j][i][0]*ImPsi[j][i][0];
            
            #Impose symmetry or antisymmetry
    		Rho = Rho + symmetry*(RePsi[j][i][0]*RePsi[i][j][1]+ImPsi[j][i][0]*ImPsi[i][j][0]);
    		
            rho_2[i] = rho_2[i] + w[k]*Rho;
            
    		k = k+1
        
        #sum 1D probabilities and print to file
        SumRho[i] = rho_1[i] + rho_2[i]
        
        if np.abs(SumRho[i]) < 1.0e-20:
            SumRho[i] = 0.0
            
        if i%10 == 0:
            None
            #fprintf(out8, "%e\n", SumRho[i]);
            
        p = p + 1 
    
    #find relative probability and print it to file 
    tmp  = np.abs((Ptot/Ptot_i)-1.0)
    
    if tmp != 0.0:
        Prel =  math.log10(tmp)
        
    #Determine 1 particle correlation function 
    #for i+j fixed (on other diagnol) vs x = i-j 
    if n == Nt/2:
        
        for i in range(1, N1+1, 5):
            j =  N1+1-i
            x = i-j
            x = np.abs(x)
            
            if (rho_1[i] != 0.0) and (rho_2[j] != 0.0):
                corr[i] =  Rho[i,j]/rho_1[i]/rho_2[j]
                
            if (rho_1[j] != 0.0) and (rho_2[i] != 0.0):
                corr[i] =  corr[i]+Rho[j,i]/rho_1[j]/rho_2[i]
                
            if corr[i] != 0.0:
                corr[i] = math.log10(np.abs(corr[i]))
                
    return 
                
#Potential
def potential():
    tmp = alpha/dx
    
    for i in range(0, N1+2):
        
        for j in range(0, N2+2):
            
            if np.abs(i - j)*dx <= alpha:
                v[i, j] = vmax
                
            else:
                v[i, j] = 0
                
    return v



#-------------------------------------------------------------------------                
#Constants#
#--------------------------------------------------------------------------   
#wavenumber particle 1
k1 = 110.0

#wavenumber particle 2
k2 = 110.0

#mass particle 1
m1 = 0.5

dm1 = 1.0/m1

#mass particle 2
m2 = 10*m1

dm2 = 1.0/m2

#time step
dt = 6.0e-8

#angular frequency
ww = (k1*k1/(2.*m1) + k2*k2/(2.*m2))*0.5*dt

#particle 1 starting position
x01 = 0.25

#particle 2 starting position
x02 = 0.75

#x dimension
xDim = 201

#space step
dx = 0.002

N1 = xDim - 2 

N2 = xDim - 2

#wave packet width parameter (Delta x)
sig = 0.05

#max potential
vmax = -49348.0

dxx2 = dx*dx

con = -1.0/(2.0*dxx2)

w = np.array([])
w = np.append(w, dx/3.0)
w = np.append(w, 4.0*dx/3.0)
w = np.append(w, 2.0*dx/3.0)

#number of time steps
Nt = 40000

dm12 = 0.5/m1

dm22 = 0.5/m2
    
dtx = dt/dxx2

con2 = (dm1+dm2)*dtx

alpha = 0.062











time.sleep(1)
end = time.time()
print(f"Runtime of the program is {end - start}")