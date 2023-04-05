#!/usr/bin/env python3
import numpy as np
from numpy import random
import pickle


# Rescaled as V = u*tau/xi
u = 0.368
v1 = u # V_+1
v2 = 0.0 # V_0
v3 = -u # V_-1
################################
## Initial condition choosing ##
################################
qm2 = -0.05 # q2/mu2
################################
A = v1*v1*0.5
qm = qm2*(1 + A) - A
sn = -0.5 # cs/cn
p = 0.0

h = 0.1 # Spatial step size.
dt = 0.0025 # Time step size.
L = 20.0
x = np.arange(-L, L, h) # Rescaled by x -> x/xi
y = np.arange(-L, L, h) # Rescaled by y -> y/xi


def Lap(v, h=h):
    # 2d Laplacian
    # Periodic along y axis
    # Neumann condition along x axis(df/dx=0)
    lx, ly = np.shape(v)
    V = np.zeros((lx + 2, ly + 2), dtype='complex') # Expanded v 
    V[1:-1, 1:-1] = v
    Lv = np.zeros_like(v, dtype='complex') # This will be returned
    # Periodic condition
    V[0, :] = V[-2, :]
    V[-1, :] = V[1, :]
    # Neumann condition
    V[:, 0] = V[:, 1]
    V[:, -1] = V[:, -2]
    Lv += (V[2:, 1:-1] + V[:-2, 1:-1] +  V[1:-1, 2:] + V[1:-1, :-2] - 4.0*V[1:-1, 1:-1])/(h*h)

    return Lv


def n(psi):
    # Density
    return psi[0]*psi[0].conj() + psi[1]*psi[1].conj() + psi[2]*psi[2].conj()


def sx(psi):
    # Spin density of x componenet
    return np.sqrt(2)*((psi[0] + psi[2])*psi[1].conj()).real


def sy(psi):
    # Spin density of y component
    return np.sqrt(2)*(psi[1]*(psi[0] - psi[2]).conj()).imag


def sz(psi):
    # Spin density z componenet
    return psi[0]*psi[0].conj() - psi[2]*psi[2].conj()


def GPE(psi):
    gpe = np.zeros_like(psi)
    
    gpe[0] = -0.5*Lap(psi[0]) + n(psi)/(1 + sn)*psi[0] - (1-qm)/(1-qm2)*psi[0] + \
             sn/(1 + sn)*(sx(psi)*psi[1]/np.sqrt(2) - 1j*sy(psi)*psi[1]/np.sqrt(2) + sz(psi)*psi[0])
    
    gpe[1] = -0.5*Lap(psi[1]) + n(psi)/(1 + sn)*psi[1] - 1/(1-qm2)*psi[1] + \
             sn/(1 + sn)*(sx(psi)*(psi[0] + psi[2])/np.sqrt(2) + 1j*sy(psi)*(psi[0] - psi[2])/np.sqrt(2))
    
    gpe[2] = -0.5*Lap(psi[2]) + n(psi)/(1 + sn)*psi[2] - (1-qm)/(1-qm2)*psi[2] + \
             sn/(1 + sn)*(sx(psi)*psi[1]/np.sqrt(2) + 1j*sy(psi)*psi[1]/np.sqrt(2) - sz(psi)*psi[2])
    
    return gpe


def ICN(psi):
    diff = 1
    tol = 1E-7
    psi_new = np.copy(psi)

    while diff > tol:
        psi_new2 = psi + dt/(2*1j)*(GPE(psi) + GPE(psi_new))
        diff = np.max(np.abs(psi_new2 - psi_new))
        psi_new = np.copy(psi_new2)
        print(diff)

    return psi_new

tag = -0.05
z1 = np.load(f'{tag}_z1.npy')
z2 = np.load(f'{tag}_z2.npy')
z3 = np.load(f'{tag}_z3.npy')
L = len(z1)

phase1 = np.exp(1j*v1*y)
phase2 = np.exp(1j*v2*y)
phase3 = np.exp(1j*v3*y)

psi = np.zeros((3, L, L), dtype='complex')
psi[0] = z1*phase1[:, np.newaxis] # m = +1
psi[1] = z2*phase2[:, np.newaxis] # m = 0
psi[2] = z3*phase3[:, np.newaxis] # m = -1

#r = np.pi/8
#psi *= np.exp(1j*(r*random.randn(3, L, L)))

psis = []
psis.append(psi)

Nt = 1000 # Total time step number

# ICN
psi_t = ICN(psi)
psis.append(psi_t)

for t in range(Nt):
    #print(psi_t)
    psi_t = ICN(psi_t)
    print(t)
    #psi_t[:,:,0] = psi_t[:,:,1]
    #psi_t[:,:,-1] = psi_t[:,:,-2]
    #psi_t[:,0,:] = psi_t[:,-1,:]
   
    #psi_t[:,:,1] = psi_t[:,:,0]
    #psi_t[:,:,-2] = psi_t[:,:,-1]
    #pbc = (psi_t[:,0,:] + psi_t[:,-1,:])/2
    #psi_t[:,0,:] = pbc
    #psi_t[:,-1,:] = pbc

    psis.append(psi_t)

    
with open('data', 'wb') as f:
    pickle.dump(psis, f)
