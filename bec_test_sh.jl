using NPZ
using Base.Threads



z1 = npzread("-0.05_z1.npy")
z2 = npzread("-0.05_z2.npy")
z3 = npzread("-0.05_z3.npy")


Lx = size(z1)[1]
Ly = Lx

h = 0.5 # Spatial step size.

l = Ly/10

u = 2*pi/l


v1 = u # V_+1
v2 = 0.0 # V_0
v3 = -u # V_-1
qm2 = -0.05 # q2/mu2
A = u*u*0.5
qm = qm2*(1 + A) - A
sn = -0.5 # cs/cn
p = 0.0


dt = 0.0025 # Time step size.


psi = zeros(ComplexF64,3,Lx,Ly)


for j in 1:Lx
    for k in 1:Ly
        psi[1,j,k] = z1[j]*exp(im*v1*k)
        psi[2,j,k] = z2[j]*exp(im*v2*k)
        psi[3,j,k] = z3[j]*exp(im*v3*k)
    end
end



function Lap(v)
    lx = size(v)[2]
    ly = size(v)[3]

    V = zeros(ComplexF64, 3, lx+2, ly+2)
    V[:, 2:end-1, 2:end-1] = v[:, :, :]
    Lv = zeros(ComplexF64, 3, lx+2, ly+2)
    # Periodic condition 
    V[:, 1, 1:end] = V[:, end-1, 1:end]
    V[:, end, 1:end] = V[:, 2, 1:end]
    # Neumann condition
    V[:, 1:end, 1] = V[:, 1:end, 2]
    V[:, 1:end, end] = V[:, 1:end, end-1]

    @threads for j = 2:size(V,2)-1
        @threads for i = 2:size(V,1)-1
                      Lv[:,i,j] = (V[:,i-1,j] + V[:,i+1,j] + V[:,i,j+1] + V[:,i,j-1] - 4.0*V[:,i,j])/(h*h)
                  end
              end

    return Lv[:, 2:end-1, 2:end-1]
end



function n(psi)
    # Density
    return psi[:,:,1].*conj(psi[:,:,1]) + psi[:,:,2].*conj(psi[:,:,2]) + psi[:,:,3].*conj(psi[:,:,3])
end


function n0(psi)
    return psi[:,:,2].*conj(psi[:,:,2])
end


function sz(psi)
    return psi[:,:,1].*conj(psi[:,:,1]) - psi[:,:,3].*conj(psi[:,:,3])
end




function GPE(x)
    y = zero(x) # x ga psi, y noon gpe
    lapx = Lap(x)
    conjx = conj(x)

    y[1,:,:] = -0.5*lapx[1,:,:] +
     1/(1 + sn)*(x[1,:,:].*conjx[1,:,:] + x[2,:,:].*conjx[2,:,:] + x[3,:,:].*conjx[3,:,:]).*x[1,:,:] -
      (1-qm)/(1-qm2).*x[1,:,:] +
       sn/(1 + sn)*((x[2,:,:].*conjx[2,:,:] + x[1,:,:].*conjx[1,:,:] - x[3,:,:].*conjx[3,:,:]).*x[1,:,:] + x[2,:,:].*x[2,:,:].*conjx[3,:,:])

    y[2,:,:] = -0.5*lapx[2,:,:] +
     1/(1 + sn)*(x[1,:,:].*conjx[1,:,:] + x[2,:,:].*conjx[2,:,:] + x[3,:,:].*conjx[3,:,:]).*x[2,:,:] -
      1/(1-qm2).*x[2,:,:] +
       sn/(1 + sn)*((x[1,:,:].*conjx[1,:,:] + x[3,:,:].*conjx[3,:,:]).*x[2,:,:] + 2*x[1,:,:].*x[3,:,:].*conjx[2,:,:])

    y[3,:,:] = -0.5*lapx[3,:,:] +
     1/(1 + sn)*(x[1,:,:].*conjx[1,:,:] + x[2,:,:].*conjx[2,:,:] + x[3,:,:].*conjx[3,:,:]).*x[3,:,:] -
      (1-qm)/(1-qm2).*x[3,:,:] +
       sn/(1 + sn)*((x[2,:,:].*conjx[2,:,:] - x[1,:,:].*conjx[1,:,:] + x[3,:,:].*conjx[3,:,:]).*x[3,:,:] + x[2,:,:].*x[2,:,:].*conjx[1,:,:])
    

    return y
end

function error(psi1, psi2)
    psi_e = sum(abs2.(psi1 - psi2))*h*h

    return psi_e
end



function ICN(psi)
    diff = 1
    tol = 1E-7
    psi_new = deepcopy(psi)

    while diff > tol
        psi_new2 = psi + dt/(2*im)*(GPE(psi) + GPE(psi_new))
        diff = error(psi_new2, psi_new)
        psi_new = deepcopy(psi_new2)
        println(diff)
    end
    return psi_new
end


psis = []

T = 200 # Total t/tau
Nt = trunc(Int64, T/0.0025) # Total time step number
println("Total time step number is ",Nt)

# ICN
psi_t = ICN(psi)
push!(psis, psi_t)

for nt = 1:Nt
    println("t/tau = $(nt*dt)")
    global psi_t = ICN(psi_t)

    if (nt % 8000) == 0
        push!(psis, psi_t)

    end
end