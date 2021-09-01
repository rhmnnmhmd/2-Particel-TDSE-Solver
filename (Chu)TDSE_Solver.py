from scipy import integrate
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
plt.rc('savefig', dpi=300)
import numpy as np
from numba import jit, njit
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

#-------------------------------------------------------------------------
#spatial separation
dx = 0.02      

#spatial grid points                 
x = np.arange(0, 10, dx)       

# wave number
kx = 0.1      

#mass                  
m = 1    

#width of initial gaussian wave-packet                      
sigma = 0.1       

#center of initial gaussian wave-packet                 
x0 = 3.0                        

#normalization constant
A = 1.0 / (sigma * np.sqrt(np.pi)) 

#Initial Wavefunction
psi0 = np.sqrt(A) * np.exp(-(x-x0)**2 / (2.0 * sigma**2)) * np.exp(1j * kx * x)

#center of V(x)
x_Vmin = 5         

#period of SHO 
T = 1           

#angular frequency
omega = 2 * np.pi / T

#wavenumber in terms of mass and angular frequency
k = omega**2 * m

#harmonic potential
V = 0.5 * k * (x - x_Vmin)**2

# Make a plot of psi0 and V 
plt.plot(x, V*0.01, "k--", label=r"$V(x) = \frac{1}{2}m\omega^2 (x-5)^2$ (x0.01)")
plt.plot(x, np.abs(psi0)**2, "r", label=r"$\vert\psi(t=0,x)\vert^2$")
plt.legend(loc=1, fontsize=8, fancybox=False)
print("Total Probability: ", np.sum(np.abs(psi0)**2)*dx)

#DEFINE LAPLACE OPERATOR---------------------------------------------------
# Laplace Operator (Finite Difference)
D2 = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(x.size, x.size)) / dx**2
#D2.toarray()*dx**2

#SOLVE TDSE----------------------------------------------------------------
#define Planck's constant
hbar = 1
#hbar = 1.0545718176461565e-34

#define dpsi/dt
def psi_t(t, psi):
    return -1j * (- 0.5 * hbar / m * D2.dot(psi) + V / hbar * psi)

#time interval for snapshots
dt = 0.005  

#initial time
t0 = 0.0  

#final time  
tf = 1.0    

#recorded time shots
t_eval = np.arange(t0, tf, dt)  

# Solve the Initial Value Problem
sol = integrate.solve_ivp(psi_t, t_span = [t0, tf], y0 = psi0, t_eval = t_eval, method="RK23")

#PLOTTING------------------------------------------------------------------
#create figure object
fig = plt.figure(figsize=(6, 4))

for i, t in enumerate(sol.t):
    #Plot Wavefunctions
    plt.plot(x, np.abs(sol.y[:,i])**2) 
    
    #Print Total Probability (Should = 1)            
    #print(np.sum(np.abs(sol.y[:,i])**2)*dx)        

#plot the potential
plt.plot(x, V * 0.01, "k--", label=r"$V(x) = \frac{1}{2}m\omega^2 (x-5)^2$ (x0.01)")   # Plot Potential

#create legend
plt.legend(loc=1, fontsize=8, fancybox=False)

#save created figure
#fig.savefig('sho@2x.png')

#ANIMATION-----------------------------------------------------------------
fig = plt.figure()
ax1 = plt.subplot(1,1,1)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 6)
title = ax1.set_title('')
line1, = ax1.plot([], [], "k--")
line2, = ax1.plot([], [])

def init():
    line1.set_data(x, V * 0.01)
    line2.set_data([], [])
    return line1, line2,

def animate(i):
    line2.set_data(x, np.abs(sol.y[:,i])**2)
    title.set_text('Time = {0:1.3f}'.format(sol.t[i]))
    return line1, line2,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(sol.t), interval=50, blit=True)

# Save the animation into a short video
anim.save('reee.mp4', fps=15, extra_args=['-vcodec', 'libx264'], dpi=600)