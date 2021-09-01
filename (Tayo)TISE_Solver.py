import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['science', 'notebook', 'grid'])

#potential energy
def Vpot(x):
    return x**2

#usr input
a = float(input('enter lower limit of the domain: '))
b = float(input('enter upper limit of the domain: '))
N = int(input('enter number of grid points: '))

#space domain
x = np.linspace(a,b,N)

#space step
h = x[1]-x[0]

#create kinetic energy matrix
T = np.zeros((N-2)**2).reshape(N-2,N-2)

for i in range(N-2):
    
    for j in range(N-2):
        
        if i==j:
            T[i,j]= -2
            
        elif np.abs(i-j)==1:
            T[i,j]=1
            
        else:
            T[i,j]=0
            
#create potential energy matrix
V = np.zeros((N-2)**2).reshape(N-2,N-2)

for i in range(N-2):
    
    for j in range(N-2):
        
        if i==j:
            V[i,j]= Vpot(x[i+1])
            
        else:
            V[i,j]=0

#create hamiltonian
H = -T/(2*h**2) + V

#solve eigenvalue problem for the Hamiltonian
val, vec = np.linalg.eig(H)

#sort eigenvalue in ascending values
z = np.argsort(val)

#take first 4 eigenvalues
z = z[0:4]
energies = val[z]/val[z, 0]
print(energies)

#plotting stuffs
plt.figure(figsize=(12,10))

for i in range(len(z)):
    y = []
    y = np.append(y, vec[: , z[i]])
    y = np.append(y, 0)
    y = np.insert(y, 0, 0)
    plt.plot(x, np.abs(y)**2, lw=3, label="{} ".format(i))
    plt.xlabel('x', size = 14)
    plt.ylabel('$\psi$(x)', size = 14)
    
plt.legend()
plt.title('normalized wavefunctions for a harmonic oscillator using finite difference method',size=14)