import numpy as np

L=100
psi=np.zeros((2,L,L), dtype=complex)
dx=0.001
dt=dx/10
g11=1
g22=1
g12=100
mu1=1
mu2=1
V1=0
V2=0
m1=1
m2=1

print(len(psi))
print(len(psi[0]))
print(len(psi[0,0]))

psi[0]=1
psi[1]=1


def devx(x) :
    y=np.zeros((len(x),len(x[0]),len(x[0,0])), dtype=complex)
    for i in range(len(x)):
        for j in range(2,len(x[0])-2):
            for k in range(0,len(x[0,0])):
                y[i,j,k]=(-x[i,j+2,k]+8*x[i,j+1,k]-8*x[i,j-1,k]+x[i,j-2,k])/(12*dx)
        for k in range(0,len(x[0,0])):
            y[i,1,k]=(x[i,2,k]-x[i,0,k])/(2*dx)
            y[i,len(x[0])-2,k]=(x[i,len(x[0])-1,k]-x[i,len(x[0])-3,k])/(2*dx)
            y[i,0,k]=(x[i,1,k]-x[i,0,k])/(dx)
            y[i,len(x[0])-1,k]=(x[i,len(x[0])-1,k]-x[i,len(x[0])-2,k])/(dx)
        
    return y


def devy(x) :
    y=np.zeros((len(x),len(x[0]),len(x[0,0])), dtype=complex)
    for i in range(len(x)):
        for j in range(0,len(x[0])):
            for k in range(2,len(x[0,0])-2):
                y[i,j,k]=(-x[i,j,k+2]+8*x[i,j,k+1]-8*x[i,j,k-1]+x[i,j,k-2])/(12*dx)
        for j in range(0,len(x[0])):
            y[i,j,1]=(x[i,j,2]-x[i,j,0])/(2*dx)
            y[i,j,len(x[0])-2]=(x[i,j,len(x[0])-1]-x[i,j,len(x[0])-3])/(2*dx)
            y[i,j,0]=(x[i,j,1]-x[i,j,0])/(dx)
            y[i,j,len(x[0])-1]=(x[i,j,len(x[0])-1]-x[i,j,len(x[0])-2])/(dx)

    return y


def dev2x(x) :
    y=np.zeros((len(x),len(x[0]),len(x[0,0])), dtype=complex)
    for i in range(len(x)):
        for j in range(1,len(x[0])-1):
            for k in range(0,len(x[0,0])):
                y[i,j,k]=(x[i,j+1,k]+x[i,j-1,k]-2*x[i,j,k])/(dx*dx)
    return y

def dev2y(x) :
    y=np.zeros((len(x),len(x[0]),len(x[0,0])), dtype=complex)
    for i in range(len(x)):
        for j in range(0,len(x[0])):
            for k in range(1,len(x[0,0])-1):
                y[i,j,k]=(x[i,j,k+1]+x[i,j,k-1]-2*x[i,j,k])/(dx*dx)
    return y


'''

def dev2x(x) :
    y=np.zeros((len(x),len(x[0]),len(x[0,0])), dtype=complex)
    for i in range(len(x)):
        for j in range(2,len(x[0])-2):
            for k in range(0,len(x[0,0])):
                y[i,j,k]=(-x[i,j+2,k]+16*x[i,j+1,k]-30*x[i,j,k]+16*x[i,j-1,k]-x[i,j-2,k])/(12*dx**2)
        for k in range(0,len(x[0,0])):
            y[i,1,k]=(x[i,2,k]+x[i,0,k]-2*x[i,1,k])/(dx**2)
            y[i,len(x[0])-2,k]=(x[i,len(x[0])-1,k]+x[i,len(x[0])-3,k]-2*x[i,len(x[0])-2,k])/(dx**2)
            y[i,0,k]=(x[i,2,k]-2*x[i,1,k]+x[i,0,k])/(dx**2)
            y[i,len(x[0])-1,k]=(x[i,len(x[0])-1,k]-2*x[i,len(x[0])-2,k]+x[i,len(x[0])-3,k])/(dx**2)
    return y

def dev2y(x) :
    y=np.zeros((len(x),len(x[0]),len(x[0,0])), dtype=complex)
    for i in range(len(x)):
        for j in range(0,len(x[0])):
            for k in range(2,len(x[0])-2):
                y[i,j,k]=(-x[i,j,k+2]+16*x[i,j,k+1]-30*x[i,j,k]+16*x[i,j,k-1]-x[i,j,k-2])/(12*dx**2)
        for j in range(0,len(x[0])):
            y[i,j,1]=(x[i,j,2]+x[i,j,0]-2*x[i,j,1])/(dx**2)
            y[i,j,len(x[0])-2]=(x[i,j,len(x[0])-1]+x[i,j,len(x[0])-3]-2*x[i,j,len(x[0])-2])/(dx**2)
            y[i,j,0]=(x[i,j,2]-2*x[i,j,1]+x[i,j,0])/(dx**2)
            y[i,j,len(x[0])-1]=(x[i,j,len(x[0])-1]-2*x[i,j,len(x[0])-2]+x[i,j,len(x[0])-3])/(dx**2)
    return y

'''

def Egp(x) :
    y=0
    
    DxX=devx(x)
    DyX=devy(x)

    for j in range(len(x[0])) :
        for k in range(len(x[0,0])) :
            y=y+(dx**2)*(1/(2*m1))*(np.absolute(DxX[0,j,k])**2+np.absolute(DyX[0,j,k])**2)
            y=y+(dx**2)*(1/(2*m2))*(np.absolute(DxX[1,j,k])**2+np.absolute(DyX[1,j,k])**2)
            y=y+(dx**2)*((V1-mu1)*np.absolute(x[0,j,k])**2+(g11/2)*np.absolute(x[0,j,k])**4+(g12/2)*np.absolute(x[0,j,k])**2*np.absolute(x[1,j,k])**2)
            y=y+(dx**2)*((V2-mu2)*np.absolute(x[1,j,k])**2+(g22/2)*np.absolute(x[1,j,k])**4+(g12/2)*np.absolute(x[0,j,k])**2*np.absolute(x[1,j,k])**2)

    return y

def lap(x) :
    return dev2x(x)+dev2y(x)


print(Egp(psi))


def evo1(x) :
    return -1j*(-(1/(2*m1))*lap(x)[0]+(V1-mu1)*x[0]+g11*(np.conjugate(x[0])*x[0])*x[0]+g12*(np.conjugate(x[1])*x[1])*x[0])


def evo2(x) :
    return -1j*(-(1/(2*m2))*lap(x)[1]+(V2-mu2)*x[1]+g22*(np.conjugate(x[1])*x[1])*x[1]+g12*(np.conjugate(x[0])*x[0])*x[1])


N=100

k1=np.zeros((len(psi),len(psi[0]),len(psi[0,0])), dtype=complex)
k2=np.zeros((len(psi),len(psi[0]),len(psi[0,0])), dtype=complex)
k3=np.zeros((len(psi),len(psi[0]),len(psi[0,0])), dtype=complex)
k4=np.zeros((len(psi),len(psi[0]),len(psi[0,0])), dtype=complex)



for t in range(N):


    k1[0]=evo1(psi)
    k1[1]=evo2(psi)

    k2[0]=evo1(psi+(dt/2)*k1)
    k2[1]=evo2(psi+(dt/2)*k1)

    k3[0]=evo1(psi+(dt/2)*k2)
    k3[1]=evo2(psi+(dt/2)*k2)

    k4[0]=evo1(psi+(dt)*k3)
    k4[1]=evo2(psi+(dt)*k3)
    
    psi=psi+dt*(k1+2*k2+2*k3+k4)/6
    
    print(Egp(psi))


print(psi)

print(np.absolute(psi))
