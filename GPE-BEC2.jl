using NPZ


u = 0.368
v1 = u # V_+1
v2 = 0.0 # V_0
v3 = -u # V_-1
qm2 = -0.05 # q2/mu2
A = u*u*0.5
qm = qm2*(1 + A) - A
sn = -0.5 # cs/cn
p = 0.0

h = 0.5 # Spatial step size.
dt = 0.0025 # Time step size.

### Initial condition ###
z1 = npzread("-0.05_z1.npy")
z2 = npzread("-0.05_z2.npy")
z3 = npzread("-0.05_z2.npy")
L = length(z1)
z1 = reshape(z1, (1, L))
z2 = reshape(z2, (1, L))
z3 = reshape(z3, (1, L))

l = 100
y = range(-l, l-h, step=h) |> collect
y = reshape(y, (1, length(y)))
#phase
s1 = exp.(im*v1*y)
s2 = exp.(im*v2*y)
s3 = exp.(im*v3*y)

psi1 = transpose(s1)*z1
psi2 = transpose(s2)*z2 
psi3 = transpose(s3)*z3
println(size(psi1))
psi = cat(psi1, psi2, psi3, dims=3)
#########################


#function Lap(v, h=h)
#    # 2d Laplacian
#    # Periodic along y axis
#    # Neumann condition along x axis(df/dx=0)
#    lx, ly = size(v)
#    
#    V = zeros(ComplexF64, lx+2, ly+2)
#    V[2:end-1, 2:end-1] = v
#    Lv = zero(v)
#    # Periodic condition 
#    V[1, 1:end] = V[end-1, 1:end]
#    V[end, 1:end] = V[2, 1:end]
#    # Neumann condition
#    V[1:end, 1] = V[1:end, 2]
#    V[1:end, end] = V[1:end, end-1]
#
#    Lv += (V[1:end-2, 2:end-1] + V[3:end, 2:end-1] + V[2:end-1, 1:end-2] + V[2:end-1, 3:end] - 4.0*V[2:end-1, 2:end-1])/(h*h)
#
#    return Lv
#end


function Lap(v, h=h)
    lx, ly = size(v)

    V = zeros(ComplexF64, lx+2, ly+2)
    V[2:end-1, 2:end-1] = v
    Lv = zeros(ComplexF64, lx+2, ly+2)
    # Periodic condition 
    V[1, 1:end] = V[end-1, 1:end]
    V[end, 1:end] = V[2, 1:end]
    # Neumann condition
    V[1:end, 1] = V[1:end, 2]
    V[1:end, end] = V[1:end, end-1]

    @threads for j = 2:size(V,2)-1
        @threads for i = 2:size(V,1)-1
                      Lv[i,j] = V[i-1,j] + V[i+1,j] + V[i,j+1] + V[i,j-1] - 4.0*V[i,j]
                  end
              end

    return Lv[2:end-1, 2:end-1]
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


function GPE(psi)
    gpe = zero(psi)

    gpe[:,:,1] = -0.5*Lap(psi[:,:,1]) + n(psi)/(1 + sn).*psi[:,:,1] - (1-qm)/(1-qm2).*psi[:,:,1] + 
                 sn/(1 + sn)*((n0(psi) + sz(psi)).*psi[:,:,1] + psi[:,:,2].*psi[:,:,2].*conj(psi[:,:,3]))

    gpe[:,:,2] = -0.5*Lap(psi[:,:,2]) + n(psi)/(1 + sn).*psi[:,:,2] - 1/(1-qm2).*psi[:,:,2] +
                 sn/(1 + sn)*((n(psi) - n0(psi)).*psi[:,:,2] + 2*psi[:,:,1].*psi[:,:,3].*conj(psi[:,:,2]))

    gpe[:,:,3] = -0.5*Lap(psi[:,:,3]) + n(psi)/(1 + sn).*psi[:,:,3] - (1-qm)/(1-qm2).*psi[:,:,3] +
                 sn/(1 + sn)*((n0(psi) - sz(psi)).*psi[:,:,3] + psi[:,:,2].*psi[:,:,2].*conj(psi[:,:,1]))

    return gpe
end


function error(psi1, psi2)
    psi_e = sum(abs2.(psi1 - psi2))*h*h

    return psi_e
end


function ICN(psi, dt=dt)
    diff = 1
    tol = 1E-7
    psi_new = copy(psi)

    while diff > tol
        psi_new2 = psi + dt/(2*im)*(GPE(psi) + GPE(psi_new))
        diff = error(psi_new2, psi_new)
        psi_new = copy(psi_new2)
        println(diff)
    end
    return psi_new
end


psis = []

T = 200 # Total t/tau
Nt = convert(Int64, T/0.0025) # Total time step number

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


for psi_i in psi_s
    psi1 = cat(psi1, psi_i[:,:,1], dims=3)
    psi2 = cat(psi2, psi_i[:,:,2], dims=3)
    psi3 = cat(psi3, psi_i[:,:,3], dims=3)

end

npzwrite("psi1.npy", psi1)
npzwrite("psi2.npy", psi2)
npzwrite("psi3.npy", psi3)
