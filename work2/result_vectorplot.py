import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import animation


h = 0.1 # Spacing
L = 20.0
x = np.arange(-L, L, h) # Rescaled by x -> x/xi
y = np.arange(-L, L, h) # Rescaled by y -> y/xi


'''
Notes:
Subscript 1, 2, 3(like z1, z2, z3) means m = +1, 0, -1 respectively.
Parameters of this code q, mu means tilde{q}, tilde{mu} of paper.

Rescaling factors
xi = hbar/sqrt(M(mu - q))
tau = hbar/(mu - q)
'''
# Spin density
def sx(psi):
    return np.sqrt(2)*((psi[0] + psi[2])*psi[1].conj()).real


def sy(psi):
    return np.sqrt(2)*(psi[1]*(psi[0] - psi[2]).conj()).imag


def sz(psi):
    return psi[0]*psi[0].conj() - psi[2]*psi[2].conj()


with open('data', 'rb') as f:
    psis = pickle.load(f)


fig1 = plt.figure()
ax = plt.gca()
psi0 = psis[0]
N = len(psis)

Sx = sx(psi0)
Sy = sy(psi0)
Sz = sz(psi0)

d = 10

X, Y = np.meshgrid(x[0:-1:d], y[0:-1:d])
Q = ax.quiver(X, Y, Sx[0:-1:d, 0:-1:d], Sy[0:-1:d, 0:-1:d], Sz[0:-1:d, 0:-1:d].real, scale=30, cmap='jet', pivot='mid')


def update(num, Q, psis):
    psi = psis[num]
    
    Sx = sx(psi)
    Sy = sy(psi)
    Sz = sz(psi)

    Q.set_UVC(Sx[0:-1:d, 0:-1:d], Sy[0:-1:d, 0:-1:d], Sz[0:-1:d, 0:-1:d].real)
    print(num)

    return Q,

anim = animation.FuncAnimation(fig1, update, fargs=(Q, psis), interval=50, frames=N, repeat=False)
anim.save('result.mp4')
