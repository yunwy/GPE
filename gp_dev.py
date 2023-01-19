import numpy as np
import scipy.constants as sc
#import scipy.sparse as ss


'''
Equation of this code is based on DOI: 10.1103/PhysRevA.85.053639 
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
dy = 0.1 # y spatial step size
dt = 0.1 # time step size
Lx = 10.0 # Total x length
Ly = 10.0 # Total y length
T = 10.0 # Total time

Nx = int(Lx/dx) # Number of points on x axis
Ny = int(Ly/dy) # Number of points on y axis
Nt = int(T/dt)  # Number of points on time axis


# External potential
def Vxy(x, y):
    m = m1
    wx = 1
    wy = 1
    
    return m*(wx*wx*x*x + wy*wy*y*y)/2
    
    
x = np.linspace(-10, 10, Nx+1)
y = np.linspace(-10, 10, Ny+1)
X, Y = np.meshgrid(x, y)

V = Vxy(X, Y)

# Chemical potential
mu1 = 1.0
mu2 = 1.0

# Initial condition: Tentative
def Psi0(x, y):
    'tentative'
    
    return 0.0
    
    
# Boundary condition: Particle in a Box. Wave function is zero at boundary
Psi = np.zeros((Nx+1, Ny+1), dtype='complex')
Psi_p = np.zeros_like(Psi, dtype='complex')

rx = 1j*sc.hbar*dt/(4*m1*dx*dx)
ry = 1j*sc.hbar*dt/(4*m1*dy*dy)


def A(psi_p):
    # Matrix for n+1 time index
    aij = (1 + 2*rx + 2*ry - 1j*dt*mu1/2)*np.ones((Nx+1, Ny+1)) + 1j*dt*V/2 - g[0,0]*psi_p*psi_p.conjugate() - g[0,1]*psi_p*psi_p.conjugate()
    
    return aij
    
    
def B(psi_p):
    # Matrix for n time index
    bij = (1 - 2*rx - 2*ry + 1j*dt*mu1/2)*np.ones((Nx+1, Ny+1)) - 1j*dt*V/2 + g[0,0]*psi_p*psi_p.conjugate() + g[0,1]*psi_p*psi_p.conjugate()
    
    return bij


sdf
