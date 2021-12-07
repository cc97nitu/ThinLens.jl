"""
    driftLinear(particle, length::Float64)

Apply linear drift to particle coordinates.
"""
function driftLinear(p::T, len::Float64) where {T<:AbstractArray{Float64}}
    pnew = Vector(p)
      
    pnew[1] += len * p[2]
    pnew[3] += len * p[4]

    return pnew
end

function ChainRulesCore.rrule(::typeof(driftLinear), pold::T, len::Float64) where {T<:AbstractArray{Float64}}
    p = driftLinear(pold, len)
    function driftLinear_pullback(Δ)
        newΔ = Vector(Δ)

        newΔ[2] += len * Δ[1]
        newΔ[4] += len * Δ[3]
        return ChainRulesCore.NoTangent(), newΔ, ChainRulesCore.NoTangent()
    end
    return p, driftLinear_pullback
end


"""
    driftExact(particle, length::Float64)

Apply analytic drift to particle coordinates.
"""
function driftExact(p::T, len::Float64) where {T<:AbstractArray{Float64}}
    pnew = Vector(p)
    px = p[2]; py = p[4]; δ = p[6]; vR = p[7]
    
    pz = sqrt((1 + δ)^2 - px^2 - py^2)
      
    pnew[1] += len * px / pz
    pnew[3] += len * py / pz
    pnew[5] += len * (1 - vR * (1+δ) / pz)

    return pnew
end

function ChainRulesCore.rrule(::typeof(driftExact), pold::T, len::Float64) where {T<:AbstractArray{Float64}}
    p = driftExact(pold, len)
    function driftExact_pullback(Δ)
        newΔ = Vector(Δ)
        px = pold[2]; py = pold[4]; δ = pold[6]; vR = pold[7]
        
        pz = sqrt((1+δ)^2 - px^2 - py^2)
        lOpz2 = len / pz^2
        
        newΔ[2] += lOpz2 * ((pz + px^2/pz) * Δ[1] + (py * Δ[3] - vR*(1+δ) * Δ[5]) * px/pz)
        newΔ[4] += lOpz2 * ((pz + py^2/pz) * Δ[3] + (px * Δ[1] - vR*(1+δ) * Δ[5]) * py/pz)
        
        pOpz = (1+δ) / pz
        newΔ[6] -= lOpz2 * (pOpz * (px * Δ[1] + py * Δ[3]) + vR * (pz - (1+δ)*pOpz) * Δ[5])
        
        newΔ[7] -= len * pOpz * Δ[5]

        return ChainRulesCore.NoTangent(), newΔ, ChainRulesCore.NoTangent()
    end
    return p, driftExact_pullback
end

"""
    tM_2ndOrder(p::T, length::Float64, kn::Vector{Float64}, ks::Vector{Float64}) where {T<:AbstractArray{Float64}}

Update particle momentum in transversal magnetic fields up to second order.
"""
function tM_2ndOrder(p::T, len::Float64, kn::Vector{Float64}, ks::Vector{Float64}) where {T<:AbstractArray{Float64}}
    pnew = Vector(p)
    knl = len .* kn; ksl = len .* ks

    dpx = knl[end]
    dpy = ksl[end]
    
    # assumes kn, ks are of same length
    for i = length(knl)-1:-1:1
        zre = (dpx * p[1] - dpy * p[3]) / i
        zim = (dpx * p[3] + dpy * p[1]) / i
        dpx = knl[i] + zre
        dpy = ksl[i] + zim
    end
    
    # update
    pnew[2] -= dpx
    pnew[4] += dpy
    
    return pnew
end

function ChainRulesCore.rrule(::typeof(tM_2ndOrder), pold::T, len::Float64, kn::Vector{Float64}, ks::Vector{Float64}) where {T<:AbstractArray{Float64}}
    pnew = tM_2ndOrder(pold, len, kn, ks)
    
    function tM_2ndOrder_pullback(grad)
        newGrad = Vector(grad)
        knl = len .* kn; ksl = len .* ks
        
        newGrad[1] += (-knl[2] - knl[3]*pold[1] + ksl[3]*pold[3]) * grad[2] + (ksl[2] + ksl[3]*pold[1] + knl[3]*pold[3]) * grad[4]
        newGrad[3] += (ksl[2] + knl[3]*pold[3] + ksl[3]*pold[1]) * grad[2] + (knl[2] - ksl[3]*pold[3] + knl[3]*pold[1]) * grad[4]
        
        Δknl = zeros(length(knl))
        Δknl[1] = -grad[2]
        Δknl[2] = -pold[1] * grad[2] + pold[3] * grad[4]
        Δknl[3] = (pold[3]^2 - pold[1]^2)/2 * grad[2] + pold[1]*pold[3] * grad[4]
        
        Δksl = zeros(length(ksl))
        Δksl[1] = grad[4]
        Δksl[2] = pold[3] * grad[2] + pold[1] * grad[4]
        Δksl[3] = pold[1]*pold[3] * grad[2] + (pold[1]^2 - pold[3]^2)/2 * grad[4]
        
        return ChainRulesCore.NoTangent(), newGrad, ChainRulesCore.NoTangent(), Δknl*len, Δksl*len
    end
    
    return pnew, tM_2ndOrder_pullback
end


"""
    curvatureEffectKick(particle, length, knl::Vector{Float64, ksl::Vector{Float64}, α::Float64, β::Float64})

Apply curvature effects to particle.
"""
function curvatureEffectKick(p::T, len::Float64, kn::Vector{Float64}, ks::Vector{Float64}, hx::Float64, hy::Float64) where {T<:AbstractArray{Float64}}
    pnew = Vector(p)
    
    hxlx = hx * len * p[1]
    hyly = hy * len * p[3]

    if len != 0.
        hxx = hxlx / len
        hyy = hyly / len
    else  # non physical weak focusing disabled (SixTrack mode)
        hxx = 0.
        hyy = 0.
    end

    dpx = len * (hx + hx * p[6] - kn[1] * hxx)
    dpy = len * (-hy - hy * p[6] + ks[1] * hyy)

    dσ = (hyly - hxlx) * p[7]
    
    # update
    pnew[2] += dpx
    pnew[4] += dpy
    pnew[5] += dσ

    # println("after curvature", " px: ", pnew[2], " py: ", pnew[4])
    return pnew
end

function ChainRulesCore.rrule(::typeof(curvatureEffectKick), pold::T, len::Float64, kn::Vector{Float64}, ks::Vector{Float64}, hx::Float64, hy::Float64) where {T<:AbstractArray{Float64}}
    pnew = curvatureEffectKick(pold, len, kn, ks, hx, hy)
    
    function curvatureEffectKick_pullback(Δ)
        newΔ = Vector(Δ)
        
        newΔ[1] -= kn[1]*hx*len * Δ[2] + hx*len*pold[7] * Δ[5]
        newΔ[2] += ks[2]*hy*len * Δ[4] + hy*len*pold[7] * Δ[5]
        newΔ[6] += hx*len * Δ[2] - hy*len * Δ[4]
        newΔ[7] += (hy*len * pold[3] - hx*len * pold[1]) * Δ[5]
        
        Δkn = zeros(length(kn))
        Δkn[1] -= hx*len * Δ[2]
        
        Δks = zeros(length(ks))
        Δks[1] += hy*len * Δ[4]
        
        return ChainRulesCore.NoTangent(), newΔ, ChainRulesCore.NoTangent(), Δkn, Δks, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
    end
    
    return pnew, curvatureEffectKick_pullback
end


"""
    dipoleEdge(p::T, curvature, edgeAngle, hgap, fint)

Kick particles due to dipole edge effects.
"""
function dipoleEdge(p::T, len::Float64, α::Float64, ϵ::Float64, hgap::Float64, fint::Float64) where {T<:AbstractArray{Float64}}
    pnew = Vector(p)
    corr = 2 * α/len * hgap * fint
    
    pnew[2] += p[1] * α/len * tan(ϵ)
    pnew[4] -= p[3] * α/len * tan(ϵ - corr / cos(ϵ) * (1 + sin(ϵ)^2))
    
    return pnew
end

function ChainRulesCore.rrule(::typeof(dipoleEdge), pold::T, len::Float64, α::Float64, ϵ::Float64, hgap::Float64, fint::Float64) where {T<:AbstractArray{Float64}}
    pnew = dipoleEdge(pold, len, α, ϵ, hgap, fint)
    
    function dipoleEdge_pullback(Δ)
        newΔ = Vector(Δ)
        corr = 2 * α/len * hgap * fint

        newΔ[1] += α/len * tan(ϵ) * Δ[2]
        newΔ[3] -= α/len * tan(ϵ - corr / cos(ϵ) * (1 + sin(ϵ)^2)) * Δ[4]
        
        return ChainRulesCore.NoTangent(), newΔ, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
    end
    
    return pnew, dipoleEdge_pullback
end

function dipoleEdge(p::T, len::Float64, α::Float64, ϵ::Float64) where {T<:AbstractArray{Float64}}
    pnew = Vector(p)
    
    pnew[2] += p[1] * α/len * tan(ϵ)
    pnew[4] -= p[3] * α/len * tan(ϵ)
    
    return pnew
end

function ChainRulesCore.rrule(::typeof(dipoleEdge), pold::T, len::Float64, α::Float64, ϵ::Float64) where {T<:AbstractArray{Float64}}
    pnew = dipoleEdge(pold, len, α, ϵ)
    
    function dipoleEdge_pullback(Δ)
        newΔ = Vector(Δ)
        
        newΔ[1] += α/len * tan(ϵ) * Δ[2]
        newΔ[3] -= α/len * tan(ϵ) * Δ[4]
        
        return ChainRulesCore.NoTangent(), newΔ, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
    end
    
    return pnew, dipoleEdge_pullback
end

"""
    thinMultipole(p::T, length::Float64, kn::Vector{Float64}, ks::Vector{Float64}) where {T<:AbstractArray{Float64}}

Update particle momentum in transversal magnetic field up to arbitrary order.
"""
function thinMultipole(p::T, len::Float64, kn::Vector{Float64}, ks::Vector{Float64}) where {T<:AbstractArray{Float64}}
    pnew = Vector(p)

    dpx = kn[end]
    dpy = ks[end]
    
    # assumes kn, ks are of same length
    for i = length(kn)-1:-1:1
        zre = (dpx * p[1] - dpy * p[3]) / i
        zim = (dpx * p[3] + dpy * p[1]) / i
        dpx = kn[i] + zre
        dpy = ks[i] + zim
    end
    
    # update
    pnew[2] -= len * dpx
    pnew[4] += len * dpy
    
    return pnew
end

function ChainRulesCore.rrule(::typeof(thinMultipole), pold::T, len::Float64, kn::Vector{Float64}, ks::Vector{Float64}) where {T<:AbstractArray{Float64}}
    pnew = thinMultipole(pold, len, kn, ks)
    
    function thinMultipole_pullback(Δ)
        newΔ = Vector(Δ)

        x = pold[1]; y = pold[3]

        # dpx/dx and dpy/dx
        δpx_δx = kn[end]
        δpy_δx = ks[end]
        for i = length(kn)-1:-1:2
            zre = (δpx_δx * x - δpy_δx * y) / (i-1)
            zim = (δpx_δx * y + δpy_δx * x) / (i-1)
            δpx_δx = zre + kn[i]
            δpy_δx = zim + ks[i]
        end

        # dpx/dy and dpy/dy
        δpx_δy = -ks[end]
        δpy_δy = kn[end]
        for i = length(kn)-1:-1:2
            zre = (δpx_δy * x - δpy_δy * y) / (i-1)
            zim = (δpx_δy * y + δpy_δy * x) / (i-1)
            δpx_δy = zre - ks[i]
            δpy_δy = zim + kn[i]
        end

        newΔ[1] -= len * (δpx_δx * Δ[2] + δpx_δy * Δ[4])
        newΔ[3] += len * (δpy_δx * Δ[2] + δpy_δy * Δ[4])

        # pullback for normal multipole strengths
        Δkn = zeros(length(kn))
        Δkn[1] = -len * Δ[2]

        δpx_δkn = 1
        δpy_δkn = 0
        for i in 2:1:length(kn)
            zre = (δpx_δkn * x - δpy_δkn * y) / (i-1)
            zim = (δpx_δkn * y + δpy_δkn * x) / (i-1)
            δpx_δkn = zre
            δpy_δkn = zim
            Δkn[i] = len * (-δpx_δkn * Δ[2] + δpy_δkn * Δ[4])
        end

        # pullback for skew multiople strengths
        Δks = zeros(length(ks))
        Δks[1] = len * Δ[4]

        δpx_δks = 0
        δpy_δks = -1
        for i = 2:1:length(ks)
            zre = (δpx_δks * x - δpy_δks * y) / (i-1)
            zim = (δpx_δks * y + δpy_δks * x) / (i-1)
            δpx_δks = zre
            δpy_δks = zim
            Δks[i] = len * (δpx_δks * Δ[2] - δpy_δks * Δ[4])
        end

        return ChainRulesCore.NoTangent(), newΔ, ChainRulesCore.NoTangent(), Δkn, Δks
    end
    
    return pnew, thinMultipole_pullback
end