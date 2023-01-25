#!/usr/bin/python3
import numpy as np
import scipy.constants as sc
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


'''
Equation of this code is based on DOI: 10.1103/PhysRevA.85.053639
i(h_bar)d(psi_j)/dt = (h_bar)^2/(2*m_j)*Laplacian(psi_j) + (Vj-muj)psi_j
                      + (g_jj*|psi_j|^2 + g_12|psi_k(k != j)|^2)*psi_j
                      
Also, code is based on below web page
https://artmenlope.github.io/solving-the-2d-schrodinger-equation-using-the-crank-nicolson-method/
'''
# Mass of each particles
m1 = 1.0
m2 = 1.0
# Inversed mass matrix
# M_I = 1/m_jk = 1/m_j + 1/m_k
M_I = np.array([[2/m1,        1/m1 + 1/m2],
                [1/m2 + 1/m2, 2/m2       ]])

# S-wave scattering lengths
a11 = 1.0 # Between same atom
a22 = 1.0 # Between same atom
a12 = 1.0 # Between different atom
a = np.array([[a11, a12],
              [a12, a22]])

# Coefficients atom-atom interction
# g_jk = 2(pi)(h_bar)^2(a_jk)/(m_jk)
g = (2*np.pi*sc.hbar**2)*a*M_I


# Parameters for Crank-Nicolson method
dx = 0.1 # x spatial step size
dy = 0.1
dt = 0.1 # time step size
Lx = 10.0 # Total x length
Ly = 10.0 # Total y length
T = 10.0 # Total time

Nx = int(Lx/dx) # Number of points on x axis
Ny = int(Ly/dy) # Number of points on y axis
Ni = (Nx-2)*(Ny-2)
Nt = int(T/dt)  # Number of points on time axis


# External potential
def Vxy(x, y, m):
    wx = 1
    wy = 1
    
    return m*(wx*wx*x*x + wy*wy*y*y)/2
    
x = np.linspace(-10, 10, Nx+1)
y = np.linspace(-10, 10, Ny+1)
X, Y = np.meshgrid(x, y)

V1 = Vxy(X, Y, m1)
V2 = Vxy(X, Y, m2)

# Chemical potential
mu1 = 1.0
mu2 = 1.0
    
# Parameters for CN_matrix
rx = 1j*sc.hbar*dt/(4*m1*dx*dx)
ry = 1j*sc.hbar*dt/(4*m1*dy*dy)


def CN_matrix(psi_1, psi_2, s=0):
    # A(B): Matrix for n+1(n) time index
    Aij = np.zeros((Ni, Ni), dtype='complex')
    Bij = np.zeros((Ni, Ni), dtype='complex')

    # Main central diagonal
    aij = (1 + 2*rx + 2*ry + 1j*dt*mu1/2)*np.ones((Nx+1, Ny+1)) + 1j*dt*V1/2 - g[s,s]*psi_p1*psi_p1.conjugate() - g[0,1]*psi_p2*psi_p2.conjugate()
    bij = (1 - 2*rx - 2*ry - 1j*dt*mu1/2)*np.ones((Nx+1, Ny+1)) - 1j*dt*V1/2 + g[s,s]*psi_p1*psi_p1.conjugate() + g[0,1]*psi_p2*psi_p2.conjugate()
    
    for k in range(Ni):
        i = 1 + k//(Ny-2)
        j = 1 + k%(Ny-2)
        
        Aij[k,k] = aij[i,j]
        Bij[k,k] = bij[i,j]
        
        if i != 1: # Lower lone diagonal
            Aij[k,(i-2)*(Ny-2)+j-1] = -ry
            Bij[k,(i-2)*(Ny-2)+j-1] = ry
            
        if i != Nx-2: # Upper lone diagonal.
            Aij[k,i*(Ny-2)+j-1] = -ry
            Bij[k,i*(Ny-2)+j-1] = ry
    
        if j != 1: # Lower main diagonal.
            Aij[k,k-1] = -rx 
            Bij[k,k-1] = rx 

        if j != Ny-2: # Upper main diagonal.
            Aij[k,k+1] = -rx
            Bij[k,k+1] = rx
            
    return Aij, Bij
    

# Initial condition: Tentative
@np.vectorize
def Psi0(x, y):
    # tentative
    
    return 0.0  
    

# 2-component BEC, 2 wavefunctions; psi1, psi2
psi1 = Psi0(X, Y) # Wavefunction initialize
# Boundary condition: Particle in a Box. Wave function is zero at boundary
psi1[0,:] = 0
psi1[-1,:] = 0
psi1[:,0] = 0
psi1[:,-1] = 0

psis1 = [] # Wavefunction storage at each discrete time
psis1.append(np.copy(psi1))


psi2 = Psi0(X, Y) # Wavefunction initialize
# Boundary condition: Particle in a Box. Wave function is zero at boundary
psi2[0,:] = 0
psi2[-1,:] = 0
psi2[:,0] = 0
psi2[:,-1] = 0

psis2 = [] # Wavefunction storage at each discrete time
psis.append(np.copy(psi2))


# Crank-Nicolson method for two wavefunctions
for i in range(1, T):
    A1, B1 = CN_matrix(psi1, psi2)
    A2, B2 = CN_matrix(psi2, psi1, s=1)
    
    Asp1 = csc_matrix(A1)
    Asp2 = csc_matrix(A2)

    psi_vec1 = psi1.reshape((Ni))
    psi_vec2 = psi2.reshape((Ni))
    
    b1 = np.matmul(B1, psi_vec1)
    b2 = np.matmul(B2, psi_vec2)
    
    psi_vec1 = spsolve(Asp1, b1)
    psi_vec2 = spsolve(Asp2, b2)
    
    psi1 = psi_vec1.reshape((Nx-2, Ny-2))
    psi2 = psi_vec2.reshape((Nx-2, Ny-2))
    
    psis1.append(np.copy(psi1))
    psis2.append(np.copy(psi2))


