import numpy as np
import numpy.random as nr
'''
Gross-Pitaevskii equation solver
Based on DOI: 10.1103/PhysRevA.105.013328

Notes:
Subscript 1, 2, 3(like z1, z2, z3) means m = +1, 0, -1 respectively.
Parameters of this code q2, mu2 means tilde{q}, tilde{mu} of paper.

Rescaling factors
xi = hbar/sqrt(M(mu - q))
'''


# Parameters
N = 1024 # Grid size

h = 0.01 # Spacing
dt = 0.025 # Time step
x = np.arange(-10.0, 10.0 + h, h) # Rescaled by x -> x/xi
y = np.arange(-10.0, 10.0 + h, h) # Rescaled by y -> y/xi
X, Y = np.meshgrid(x, y)

# Rescaled as V -> V*tau/xi
v1 = 0.368 # V_+1
v2 = 0.0 # V_0
v3 = -0.368 # V_-1
################################
## Initial condition choosing ##
################################
qm2 = -0.25 # q2/mu2
################################
a = v1*v1*0.5
qm = qm2*(1 + a) - a
sn = -0.5 # cs/cn
p = 0.0

'''
# Trap potential for m=+1
def POT1(x, y):
    return 0.0
    
  
# Trap potential for m=0
def POT2(x, y):
    return 0.0
    

# Trap potential for m=-1
def POT3(x, y):
    return 0.0
''' 

# Spin-1 matrix, h_bar omitted
sigma_x = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])/np.sqrt(2)
sigma_y = 1j*np.array([[0, -1, 0], [1, 0, -1], [0, 1, 0]])/np.sqrt(2)
sigma_z = np.diag([1, 0 ,-1])
sigma_z2 = np.diag([1, 0, 1]) # (sigma_z)^2, np.dot(sigma_z, sigma_z)


# Laplacian in 2D
def Lap(v):
    Lv = np.zeros_like(v)
    # Inner: Central
    Lv[1:-1, 1:-1] = ((v[:-2, 1:-1] + v[1:-1, :-2] - 4.0*v[1:-1, 1:-1] + v[2:, 1:-1] + v[1:-1, 2:])/(h*h))
    # Up edge: y-Forward, x-Central
    Lv[0, 1:-1] = (2.0*v[0, 1:-1] - 5.0*v[1, 1:-1] + 4.0*v[2, 1:-1] - v[3, 1:-1])/(h*h)
    Lv[0, 1:-1] += (v[0, 2:] - 2.0*v[0, 1:-1] + v[0, :-2])/(h*h)
    # Down edge: y-Backward, x-Central
    Lv[-1, 1:-1] = (2.0*v[-1, 1:-1] - 5.0*v[-2, 1:-1] + 4.0*v[-3, 1:-1] - v[-4, 1:-1])/(h*h)
    Lv[-1, 1:-1] += (v[-1, 2:] - 2.0*v[-1, 1:-1] + v[-1, :-2])/(h*h)
    # Left edge: x-Forward, y-Central
    Lv[1:-1, 0] = (2.0*v[1:-1, 0] - 5.0*v[1:-1, 1] + 4.0*v[1:-1, 2] - v[1:-1, 3])/(h*h)
    Lv[1:-1, 0] += (v[2:, 0] - 2.0*v[1:-1, 0] + v[:-2, 0])/(h*h)
    # Right edge: x-Backward, y-Central
    Lv[1:-1, -1] = (2.0*v[1:-1, -1] - 5.0*v[1:-1, -2] + 4.0*v[1:-1, -3] - v[1:-1, -4])/(h*h)
    Lv[1:-1, -1] += (v[2:, -1] - 2.0*v[1:-1, -1] + v[:-2, -1])/(h*h)
    # Left-up Corner: x-Forward, y-Forward
    Lv[0, 0] = (2.0*v[0, 0] - 5.0*v[0, 1]  + 4.0*v[0, 2]  - v[0, 3])/(h*h)
    Lv[0, 0] += (2.0*v[0, 0] - 5.0*v[1, 0]  + 4.0*v[2, 0]  - v[3, 0])/(h*h)
    # Right-up Corner: x-Backward, y-Forward
    Lv[0, -1] = (2.0*v[0, -1] - 5.0*v[0, -2]  + 4.0*v[0, -3]  - v[0, -4])/(h*h)
    Lv[0, -1] += (2.0*v[0, -1] - 5.0*v[1, -1]  + 4.0*v[2, -1]  - v[3, -1])/(h*h)
    # Left-down Corner: x-Forward, y-Backward
    Lv[-1, 0] = (2.0*v[-1, 0] - 5.0*v[-1, 1]  + 4.0*v[-1, 2]  - v[-1, 3])/(h*h)
    Lv[-1, 0] += (2.0*v[-1, 0] - 5.0*v[-2, 0]  + 4.0*v[-3, 0]  - v[-4, 0])/(h*h)
    # Right-down Corner: x-Backward,y-Backward
    Lv[-1, -1] = (2.0*v[-1, -1] - 5.0*v[-1, -2]  + 4.0*v[-1, -3]  - v[-1, -4])/(h*h)
    Lv[-1, -1] += (2.0*v[-1, -1] - 5.0*v[-2, -1]  + 4.0*v[-3, -1]  - v[-4, -1])/(h*h)

    return Lv


# Density
def n(f1, f2, f3):
    return f1.conj()*f1 + f2.conj()*f2 + f3.conj()*f3


# Spin density
# x component
def sx(f1, f2, f3):
    return np.sqrt(2)*((f1 + f3)*f2.conj()).real


# y component
def sy(f1, f2, f3):
    return np.sqrt(2)*(f2*((f1 - f3).conj())).imag


# z component
def sz(f1, f3):
    return f1.conj()*f1 - f3.conj()*f3


# Initial condition: Domain wall
z1 = np.load(f'{qm2}_z1.npy')
z2 = np.load(f'{qm2}_z2.npy')
z3 = np.load(f'{qm2}_z3.npy')

phase1 = np.exp(1j*v1*y)
phase2 = np.exp(1j*v2*y)
phase3 = np.exp(1j*v3*y)

psi1 = z1*phase1[:, np.newaxis]
psi2 = z2*phase2[:, np.newaxis]
psi3 = z3*phase3[:, np.newaxis]

psi10 = np.copy(psi1)
psi20 = np.copy(psi2)
psi30 = np.copy(psi3)

# Random fluctuation
# cf) https://codetorial.net/numpy/random.html
psi1 += nr.rand(np.shape(psi1)[0], np.shape(psi)[1])
psi2 += nr.rand(np.shape(psi2)[0], np.shape(ps2)[1])
psi3 += nr.rand(np.shape(psi3)[0], np.shape(ps3)[1])

# Boundary condition
'''
Periodic along y axis; refer to the paper
Neumann condition(df/dx = 0) along x axis; We are interested in domain wall in the middle
'''

# Time evolution by Gross-Pitaevskii equation
'''
There is a term of vec{s}^2 and its derivative is 2*vec{s}@∂vec{s}/∂(ψ*_m). Here, @ means dot product.
Each derivative is ∂vec{s}/∂(ψ*_+1), ∂vec{s}/∂(ψ*_0), ∂vec{s}/∂(ψ*_-1) respectively.
∂vec{s}/∂(ψ*_+1) = (ψ_0/sqrt(2), -iψ_0/sqrt(2), ψ_+1)
∂vec{s}/∂(ψ*_0) = ((ψ_+1 + ψ_-1)/sqrt(2), i(ψ_+1 - ψ_-1)/sqrt(2), 0)
∂vec{s}/∂(ψ*_-1) = (ψ_0/sqrt(2), iψ_0/sqrt(2), -ψ_-1)

Also, There is s_z term and its derivative ∂vec{s_z}/∂(ψ*_m) is just m*ψ_m
∂s_z/∂(ψ*_+1) = ψ_+1
∂s_z/∂(ψ*_0) = 0
∂s_z/∂(ψ*_-1) = -ψ_-1

Sigma-z-sqaure term, the lastest term of this equation, is more simple.
∂(Sigma-z-sqaure)/∂(ψ*_+1) = ψ*_+1
∂(Sigma-z-sqaure)/∂(ψ*_0) = 0
∂(Sigma-z-sqaure)/∂(ψ*_-1) = ψ*_-1
'''

def GPE1(psi1, psi2, psi3):
    # m = +1
    S1 = sx(psi1, psi2, psi3)*psi2/np.sqrt(2) - 1j*sy(psi1, psi2, psi3)*psi2/np.sqrt(2) + \
         sz(psi1, psi3)*psi1 # Spin term
    
    gpe = -0.5*Lap(psi1) + sn/(1 + sn)*S1 - (qm)/(1 - qm2)*psi1# - p/(mu2 - q2)*psi1

    return -1j*gpe


def GPE2(psi1, psi2, psi3):
    # m = 0
    S2 = sx(psi1, psi2, psi3)*(psi1 + psi3)/np.sqrt(2) + 1j*sy(psi1, psi2, psi3)*(psi1 - psi3)/np.sqrt(2) # Spin term
    
    gpe = -0.5*Lap(psi2)+ sn/(1 + sn)*S2 - 1/(1 - qm2)*psi2

    return -1j*gpe


def GPE3(psi1, psi2, psi3):
    # m = -1
    S3 = sx(psi1, psi2, psi3)*psi2/np.sqrt(2) + 1j*sy(psi1, psi2, psi3)*psi2/np.sqrt(2) - \
         sz(psi1, psi3)*psi3 # Spin term
    
    gpe = -0.5*Lap(psi3) + sn/(1 + sn)*S3 - (qm)/(1 - qm2)*psi3# + p/(mu2 - q2)*psi3

    return -1j*gpe


def RK4(f1, f2, f3, x1, x2, x3, dt):
    k11 = dt*f1(x1, x2, x3)
    k12 = dt*f2(x1, x2, x3)
    k13 = dt*f3(x1, x2, x3)
    
    k21 = dt*f1(x1 + 0.5*k11, x2 + 0.5*k12, x3 + 0.5*k13)
    k22 = dt*f2(x1 + 0.5*k11, x2 + 0.5*k12, x3 + 0.5*k13)
    k23 = dt*f3(x1 + 0.5*k11, x2 + 0.5*k12, x3 + 0.5*k13)
    
    k31 = dt*f1(x1 + 0.5*k21, x2 + 0.5*k22, x3 + 0.5*k23)
    k32 = dt*f2(x1 + 0.5*k21, x2 + 0.5*k22, x3 + 0.5*k23)
    k33 = dt*f3(x1 + 0.5*k21, x2 + 0.5*k22, x3 + 0.5*k23)
    
    k41 = dt*f1(x1 + k31, x2+k32, x3 + k33)
    k42 = dt*f2(x1 + k31, x2+k32, x3 + k33)
    k43 = dt*f3(x1 + k31, x2+k32, x3 + k33)

    x1_new = x1 + (k11 + 2.0*k21 + 2.0*k31 + k41)/6.0
    x2_new = x2 + (k12 + 2.0*k22 + 2.0*k32 + k42)/6.0
    x3_new = x3 + (k13 + 2.0*k23 + 2.0*k33 + k43)/6.0
    
    return x1_new, x2_new, x3_new


for i in range(10):
    psi1_new, psi2_new, psi3_new = RK4(GPE1, GPE2, GPE3, psi1, psi2, psi3, dt)

    psi1 = psi1_new/np.sqrt(psi1_new.conjugate()*psi1_new+psi2_new.conjugate()*psi2_new+psi3_new.conjugate()*psi3_new)
    psi2 = psi2_new/np.sqrt(psi1_new.conjugate()*psi1_new+psi2_new.conjugate()*psi2_new+psi3_new.conjugate()*psi3_new)
    psi3 = psi3_new/np.sqrt(psi1_new.conjugate()*psi1_new+psi2_new.conjugate()*psi2_new+psi3_new.conjugate()*psi3_new)
    
    print(i)
    
