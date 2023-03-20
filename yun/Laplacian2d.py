import numpy as np
import matplotlib.pyplot as plt


h = 0.01
x = np.arange(-6.0, 6.0 + h, h)
y = np.arange(-6.0, 6.0 + h, h)
X, Y = np.meshgrid(x, y)


def v(x, y):
    return x*x*y*y


def Lv_exact(x, y):
    return 2*(x*x + y*y)


def L(v):
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


V = v(X, Y)

Lv1 = Lv_exact(X, Y)
Lv2 = L(V)

fig = plt.figure()

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(X, Y, Lv1, cmap='jet')

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(X, Y, Lv2, cmap='jet')

print(np.max(np.abs(Lv1 - Lv2)))

plt.show()
