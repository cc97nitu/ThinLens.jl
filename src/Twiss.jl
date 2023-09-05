module Twiss

import DSP
import LinearAlgebra as LA
import TaylorSeries as TS
import ThinLens as TL

function linearNormalForm_pseudo6d(M::S) where S<:AbstractArray
    @assert size(M) == (6,6) "M needs to be 6x6 matrix"
    
    M_pseudo = copy(M)
    ν_s = 2π/10  # fake synchrotron oscillation
    M_pseudo[5:6,5:6] = [cos(ν_s) sin(ν_s); -sin(ν_s) cos(ν_s)]
    
    vals, V = LA.eigen(M_pseudo)
    
    @assert all(vals |> real .|> abs .< 1.) && all(vals |> imag .!= 0.) "unstable modes detected"
    
    # clockwise modes
    modes = collect(1:6)[vals .|> angle .< 0.]
    
    n1 = real(V[1,modes[1]]) * imag(V[2,modes[1]]) |> abs |> sqrt
    n2 = real(V[3,modes[2]]) * imag(V[4,modes[2]]) |> abs |> sqrt
    n3 = real(V[5,modes[3]]) * imag(V[6,modes[3]]) |> abs |> sqrt

    T = hcat(
        (V[:,modes[1]] .|> real) ./ -n1, (V[:,modes[1]] .|> imag) ./ n1,
        (V[:,modes[2]] .|> real) ./ -n2, (V[:,modes[2]] .|> imag) ./ n2,
        (V[:,modes[3]] .|> real) ./ -n3, (V[:,modes[3]] .|> imag) ./ n3,
    )
    
    # remove pseudo synchrotron oscillations
    T[5:end,:] .= 0.
    T[:,5:end] .= 0.
    T[5,5] = 1.
    T[6,6] = 1.
    
    # set dispersion terms
    dx_dpx_dy_dpy = -1 * LA.inv(M[1:4, 1:4] - LA.I) * M[1:4,6]
    T[1:4,6] = dx_dpx_dy_dpy
    
    # set crab terms
    dx_dpx_dy_dpy = -1 * LA.inv(M[1:4, 1:4] - LA.I) * M[1:4,5]
    T[1:4,5] = dx_dpx_dy_dpy
    
    return T 
end


function rotate_all!(W)
    ϕ1 = atan.(W[1,2,:],W[1,1,:])
    ϕ2 = atan.(W[3,4,:],W[3,3,:])
    ϕ3 = atan.(W[5,6,:],W[5,5,:])
    
    v1 = W[:,1,:] + 1im * W[:,2,:]
    v2 = W[:,3,:] + 1im * W[:,4,:]
    v3 = W[:,5,:] + 1im * W[:,6,:]
    
    for i in 1:6
        v1[i,:] .*= exp.(-1im * ϕ1)
        v2[i,:] .*= exp.(-1im * ϕ2)
        v3[i,:] .*= exp.(-1im * ϕ3)
    end
    
    W[:,1,:] = real.(v1)
    W[:,2,:] = imag.(v1)
    W[:,3,:] = real.(v2)
    W[:,4,:] = imag.(v2)
    W[:,5,:] = real.(v3)
    W[:,6,:] = imag.(v3)
    
    return ϕ1, ϕ2
end


function periodic(model, beam)
    old_vars = TS.get_variable_names()
    old_order = TS.get_order()

    # determine one-turn map
    origin = TS.set_variables("x a y b σ δ β0β", order=1)

    otm = model(origin) |> TL.jacobian .|> TS.constant_term
    otm = otm[1:6,1:6]
    
    # linear normal form
    W = linearNormalForm_pseudo6d(otm)
    
    # numerical differentiation along circumference
    scale_eigen = 1e-4
    stencil = zeros(7,13)
    for c in 1:6
        stencil[c,:] = vcat(0., -scale_eigen * W[c,:], scale_eigen * W[c,:])
    end

    TL.setVelocityRatio!(stencil, beam)
        
    trckd = Array{Float64}(undef, 7, 13, length(model))
    trckd[:,:,1] = model[1](stencil)
    for (i,cell) in enumerate(model[2:end])
        trckd[:,:,i+1] = cell(trckd[:,:,i])
    end
    
    W_s = 1/2 * trckd[1:6,2:7,:] / scale_eigen
    W_s -= 1/2 * trckd[1:6,8:end,:] / scale_eigen

    ψx,ψy = rotate_all!(W_s)
    ψx = DSP.Unwrap.unwrap!(ψx)
    ψy = DSP.Unwrap.unwrap!(ψy)
    
    # dispersion
    dx = @. (W_s[1,6,:] - W_s[1,5,:] * W_s[5,6,:] / W_s[5,5,:]) / (W_s[6,6,:] - W_s[6,5,:] * W_s[5,6,:] / W_s[5,5,:])
    dpx = @. (W_s[2,6,:] - W_s[2,5,:] * W_s[5,6,:] / W_s[5,5,:]) / (W_s[6,6,:] - W_s[6,5,:] * W_s[5,6,:] / W_s[5,5,:])
    
    # restore previous TS setting
    TS.set_variables(join(old_vars, " "), order=old_order)

    # twiss
    return Dict(:W_s => W_s,
        :βx => W_s[1,1,:].^2, :βy => W_s[3,3,:].^2,
        :αx => -W_s[2,1,:] .* W_s[1,1,:].^2, :αy => -W_s[4,3,:] .* W_s[3,3,:].^2,
        :μx => ψx, :μy => ψy, :q1 => ψx[end]/(2π), :q2 => ψy[end]/(2π),
        :dx => dx, :dpx => dpx,
    )
end


end # module
