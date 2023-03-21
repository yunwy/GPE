import numpy as np
import matplotlib.pyplot as plt


'''
Domain wall equation using Arrested Newton Flow method.
Based on DOI: 10.1103/PhysRevA.105.013328

Notes:
Subscript 1, 2, 3(like z1, z2, z3) means m = +1, 0, -1 respectively.
Parameters of this code q, mu means tilde{q#!/usr/bin/env python3
import numpy as np
#import matplotlib.pyplot as plt


'''
Domain wall equation using Arrested Newton Flow method.
Based on DOI: 10.1103/PhysRevA.105.013328

Notes:
Subscript 1, 2, 3(like z1, z2, z3) means m = +1, 0, -1 respectively.
Parameters of this code q, mu means tilde{q}, tilde{mu} of paper.

Rescaling factors
xi = hbar/sqrt(M(mu - q))
'''

# Parameters
dy = 0.01
y = np.arange(-10.0, 10.0 + dy, dy) # Rescaled length by y = x/xi, xi = hbar/sqrt(M(mu-q))

qm = -0.05 # q/mu, variable
sn = -0.5 # cs/cn, constant


def sz(f1, f3):
    # z component of spin density
    return f1*f1 - f3*f3


def s2(f1, f2, f3):
    # Square of magnitude of spin density
    return  2.0*f2*f2*(f1 + f3)*(f1 + f3) + (f1*f1 - f3*f3)*(f1*f1 - f3*f3)


def n(f1, f2, f3):
    # Density
    return f1*f1 + f2*f2 + f3*f3


def Upot(f1, f2, f3):
    # Potential functional, scaled as U -> U/((mu - q)sqrt(n_F))
    return 0.5/(1.0+sn)*n(f1, f2, f3)*n(f1, f2, f3) - n(f1, f2, f3)/(1.0-qm) + qm/(1.0-qm)*(z1*z1+z3*z3) + \
           0.5*sn/(1.0+sn)*s2(f1, f2, f3)


def ddf(f):
    # Second derivative of arbitrary function f (1D laplacian)
    # Accuracy: 2
    Lf = np.zeros_like(f)
    Lf[1:-1] = (f[:-2] - 2.0*f[1:-1] + f[2:])/(dy*dy) # Inner: Central
    Lf[0] = (2.0*f[0] - 5.0*f[1] + 4.0*f[2] - f[3])/(dy*dy) # Left edge: Forward
    Lf[-1] = (2.0*f[-1] - 5.0*f[-2] + 4.0*f[-3] - f[-4])/(dy*dy) # Right edge: Backward
    
    return Lf


def E(f1, f2, f3):
    # Total energy functional, scaled as G -> G/((mu - q)sqrt(n_F))
    return np.sum(-0.5*(f1*ddf(f1) + f2*ddf(f2) + f3*ddf(f3))*dy + Upot(f1, f2, f3)*dy) # Naively integrate


def dE1(f1, f2, f3):
    # Force for m = +1
    return -ddf(f1) + 2.0/(1 + sn)*n(f1, f2, f3)*f1 + 2.0*sn/(1 + sn)*sz(f1, f3)*f1 + \
           -2.0*f1 + 2.0*sn/(1 + sn)*(f1 + f3)*f2*f2


def dE2(f1, f2, f3):
    # Force for m = 0
    return -ddf(f2) - 2.0/(1 - qm)*f2 + 2.0/(1 + sn)*n(f1, f2, f3)*f2 + 2.0*sn/(1 + sn)*(f1 + f3)*(f1 + f3)*f2


def dE3(f1, f2, f3):
    # Force for m = -1
    return -ddf(f3) + 2.0/(1 + sn)*n(f1, f2, f3)*f3 - 2.0*sn/(1 + sn)*sz(f1, f3)*f3 + \
           -2.0*f3 + 2.0*sn/(1 + sn)*(f1 + f3)*f2*f2


# Initial guessing, zm = fm/sqrt(n_F)
z1 = -y+0.2
z2 = np.exp(-y**2)
z3 = y + 0.2

# Initial virtual velocities for Arrested Newton's method
dz1 = np.zeros_like(y)
dz2 = np.zeros_like(y)
dz3 = np.zeros_like(y)

dz0 = np.zeros_like(y) # For arrest

dt = 0.005 # Virtual time interval
tol = 1e-10 # Difference tolerance
diff = 1.0 # Difference, initial
it = 0 # Current iterarion number
max_it = 5000 # Maximum iteration number


for t in range(max_it):
    E1 = E(z1, z2, z3)

    z1 += dt*dz1
    z2 += dt*dz2
    z3 += dt*dz3

    z1[0] = z1[1]
    z1[-1] = z1[-2]
    z2[0] = z2[1]
    z2[-1] = z2[-2]
    z3[0] = z3[1]
    z3[-1] = z3[-2]

    E2 = E(z1, z2, z3)

    print(f'{E1} -> {E2}')
    diff = E2 - E1

    if E2 > E1:
        print('Arrest')
        dz1 = np.copy(dz0)
        dz2 = np.copy(dz0)
        dz3 = np.copy(dz0)

    else:
        dz1 -= dt*dE1(z1, z2, z3)
        dz2 -= dt*dE2(z1, z2, z3)
        dz3 -= dt*dE3(z1, z2, z3)


np.save(f'{qm}_z1', z1)
np.save(f'{qm}_z2', z2)
np.save(f'{qm}_z3', z3)
}, tilde{mu} of paper.

Rescaling factors
xi = hbar/sqrt(M(mu - q))
'''

# Parameters
dy = 0.01
y = np.arange(-10.0, 10.0 + dy, dy) # Rescaled length by y = x/xi, xi = hbar/sqrt(M(mu-q))

qm = -0.05 # q/mu, variable
sn = -0.5 # cs/cn, constant

tol = 1e-10 # Difference tolerance
diff = 1.0 # Difference, initial
it = 0 # Current iterarion number
max_it = 5000 # Maximum iteration number


def sz(f1, f3):
    # z component of spin density
    return f1*f1 - f3*f3


def s2(f1, f2, f3):
    # Square of magnitude of spin density
    return  2.0*f2*f2*(f1 + f3)*(f1 + f3) + (f1*f1 - f3*f3)*(f1*f1 - f3*f3)


def n(f1, f2, f3):
    # Density
    return f1*f1 + f2*f2 + f3*f3


def Upot(f1, f2, f3):
    # Potential functional, scaled as U -> U/((mu - q)sqrt(n_F))
    return 0.5/(1.0+sn)*n(f1, f2, f3)*n(f1, f2, f3) - n(f1, f2, f3)/(1.0-qm) + qm/(1.0-qm)*(z1*z1+z3*z3) + \
           0.5*sn/(1.0+sn)*s2(f1, f2, f3)


def ddf(f):
    # Second derivative of arbitrary function f (1D laplacian)
    # Accuracy: 2
    Lf = np.zeros_like(f)
    Lf[1:-1] = (f[:-2] - 2.0*f[1:-1] + f[2:])/(dy*dy) # Inner: Central
    Lf[0] = (2.0*f[0] - 5.0*f[1] + 4.0*f[2] - f[3])/(dy*dy) # Left edge: Forward
    Lf[-1] = (2.0*f[-1] - 5.0*f[-2] + 4.0*f[-3] - f[-4])/(dy*dy) # Right edge: Backward
    
    return Lf


def E(f1, f2, f3):
    # Total energy functional, scaled as G -> G/((mu - q)sqrt(n_F))
    return np.sum(-0.5*(f1*ddf(f1) + f2*ddf(f2) + f3*ddf(f3))*dy + Upot(f1, f2, f3)*dy) # Naively integrate


def dE1(f1, f2, f3):
    # Force for m = +1
    return -ddf(f1) + 2.0/(1 + sn)*n(f1, f2, f3)*f1 + 2.0*sn/(1 + sn)*sz(f1, f3)*f1 + \
           -2.0*f1 + 2.0*sn/(1 + sn)*(f1 + f3)*f2*f2


def dE2(f1, f2, f3):
    # Force for m = 0
    return -ddf(f2) - 2.0/(1 - qm)*f2 + 2.0/(1 + sn)*n(f1, f2, f3)*f2 + 2.0*sn/(1 + sn)*(f1 + f3)*(f1 + f3)*f2


def dE3(f1, f2, f3):
    # Force for m = -1
    return -ddf(f3) + 2.0/(1 + sn)*n(f1, f2, f3)*f3 - 2.0*sn/(1 + sn)*sz(f1, f3)*f3 + \
           -2.0*f3 + 2.0*sn/(1 + sn)*(f1 + f3)*f2*f2


# Initial guessing, zm = fm/sqrt(n_F)
z1 = -y+0.2
z2 = np.exp(-y**2)
z3 = y + 0.2

# Initial virtual velocities for Arrested Newton's method
dz1 = np.zeros_like(y)
dz2 = np.zeros_like(y)
dz3 = np.zeros_like(y)

dz0 = np.zeros_like(y) # For arrest

dt = 0.005 # Virtual time interval
max_it = 15000 # Maximum iteration number
diff = 1.0 # Current differece
tol = 1e-4 # Difference tolerance


for t in range(max_it):
    E1 = E(z1, z2, z3)

    z1 += dt*dz1
    z2 += dt*dz2
    z3 += dt*dz3

    z1[0] = z1[1]
    z1[-1] = z1[-2]
    z2[0] = z2[1]
    z2[-1] = z2[-2]
    z3[0] = z3[1]
    z3[-1] = z3[-2]

    E2 = E(z1, z2, z3)

    print(f'{E1} -> {E2}')
    diff = E2 - E1

    if E2 > E1:
        print('Arrest')
        dz1 = np.copy(dz0)
        dz2 = np.copy(dz0)
        dz3 = np.copy(dz0)

    else:
        dz1 -= dt*dE1(z1, z2, z3)
        dz2 -= dt*dE2(z1, z2, z3)
        dz3 -= dt*dE3(z1, z2, z3)


np.save(f'{qm}_z1', z1)
np.save(f'{qm}_z2', z2)
np.save(f'{qm}_z3', z3)
'''
plt.plot(y, z1**2, label='m=+1')
plt.plot(y, z2**2, label='m=0')
plt.plot(y, z3**2, label='m=-1')
plt.grid()
plt.legend()
plt.title(f'q/mu={qm}')
plt.show()
'''
