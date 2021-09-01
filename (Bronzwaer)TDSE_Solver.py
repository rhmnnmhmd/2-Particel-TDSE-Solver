# This Python script integrates the time-dependent Schrodinger equation 
# using a 4th-order Runge-Kutta algorithm and plots an animation of the
# result.
# To save the animation, install FFMPEG and uncomment lines 12, 99, and 100.
# Author: Thomas Bronzwaer
# Date: 10 Aug. 2020
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cmath
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
 
# NUMERICAL METHODS
###################
 
def get_norm(psi):
    norm = 0.0
    
    for i in range(Npoints):
        norm = norm + dx * abs(psi[i])**2
        
    return norm
 
# Return a complex vector containing the second spatial derivative of psi.
def second_deriv(psi):
    secondderiv = np.zeros(Npoints, dtype=complex)
    
    for i in range(1, Npoints - 1):
        secondderiv[i] = (psi[i+1] + psi[i-1] - 2. * psi[i]) / (dx * dx)
    
    secondderiv[0] = (psi[1] + psi[Npoints-1] - 2. * psi[0]) / (dx * dx)  
    secondderiv[Npoints-1] = (psi[0] + psi[Npoints-2] - 2. * psi[Npoints-1]) / (dx * dx)   
    
    return secondderiv
 
# Update psi using a single RK4 step with timestep dt.
def update_psi_rk4(psi, dt):
    k1 = -(1. / (2.j)) * second_deriv(psi)
    k2 = -(1. / (2.j)) * second_deriv(psi + 0.5 * dt * k1)
    k3 = -(1. / (2.j)) * second_deriv(psi + 0.5 * dt * k2)
    k4 = -(1. / (2.j)) * second_deriv(psi + dt * k3)
     
    psinew = psi + dt * 1. / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    
    return psinew
 
# SIMULATION PARAMETERS
#######################
 
L         = 1. # Domain length is 2L. Specified in units of a_0.
Npoints   = 201   
sigma     = 1./12.
x         = np.linspace(-L, L, Npoints)
dx        = x[1]-x[0]
time_unit = 2.4188843265857e-17
timestep  = 0.0001
psi       = np.zeros(Npoints, dtype=complex)
 
# Initialize and normalize psi.
for i in range(Npoints):
    psi[i] = np.exp(-x[i]**2 / 2 / sigma / sigma) 
    
norm = get_norm(psi)
psi = psi/np.sqrt(norm)
    
# Set up figure.
fig, ax = plt.subplots()
line, = ax.plot(x, psi.real)
plt.ylim(-.0,abs(np.max(psi))**2 * 1.05)
plt.xlim(-1.,1.)
plt.xlabel(r'$x$ [$a_0$]')
plt.ylabel(r'$\left| \psi \right|^2$')
textheight = abs(np.max(psi))**2
steptext = ax.text(-0.98, textheight * 1, 'Integration step: 0')
timetext = ax.text(-0.98, textheight * 0.95, 'Elapsed time [s]: 0')
normtext = ax.text(-0.98, textheight * 0.9, 'Norm: 0.99999999999')
plt.title(r'Time evolution of Gaussian $\psi_0$, periodic boundary conditions')
 
rk4_steps_per_frame = 4
 
# ANIMATION FUNCTIONS
#####################
 
def animate(i):
    global psi
    
    for q in range(rk4_steps_per_frame):
        psinew = update_psi_rk4(psi, timestep)
        psi = psinew
        
    currentnorm = get_norm(psi)
    line.set_ydata(abs(psi)**2)  # update the data
    steptext.set_text('Integration step: %s'%(i * rk4_steps_per_frame))
    timetext.set_text('Elapsed time [s]: %.3e'%(i * rk4_steps_per_frame * timestep * time_unit))
    normtext.set_text('Norm: %.15s'%(currentnorm))
    
    return line,
 
def init():
    return line,
 
# RUN ANIMATION AND SAVE OR SHOW
################################
ani = animation.FuncAnimation(fig, animate, np.arange(1, 1620), init_func=init,
                              interval=25, save_count=1620)
 
FFwriter = animation.FFMpegWriter(fps=60, extra_args=['-vcodec', 'libx264'])
ani.save("/Users/ACER/Desktop/Dirichlet_BC_.mp4", writer = FFwriter)
 
plt.show()