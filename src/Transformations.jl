"""
    driftLinear(particle, length::Float64)

Apply linear drift to particle coordinates.
"""
function driftLinear(x::T, px::T, y::T, py::T, σ::T, δ::T, β0β::T, len::AbstractFloat) where T<:AbstractVector
    newx = @. x + len * px
    newy = @. y + len * py
    return newx, px, newy, py, σ, δ, β0β
end

function ChainRulesCore.rrule(::typeof(driftLinear), x::T, px::T, y::T, py::T, σ::T, δ::T, β0β::T, len::AbstractFloat) where T<:AbstractVector
    p = driftLinear(x, px, y, py, σ, δ, β0β, len)
    function driftLinear_pullback(Δ)
        Δx, Δpx, Δy, Δpy, Δσ, Δδ, Δβ0β = Δ

        newΔpx = @. Δpx + len * Δx
        newΔpy = @. Δpy + len * Δy
        return ChainRulesCore.NoTangent(), Δx, newΔpx, Δy, newΔpy, Δσ, Δδ, Δβ0β, ChainRulesCore.NoTangent()
    end
    return p, driftLinear_pullback
end


"""
    driftExact(particle, length::Float64)

Apply analytic drift to particle coordinates.
"""
function driftExact(x::T, px::T, y::T, py::T, σ::T, δ::T, β0β::T, len::AbstractFloat) where T<:AbstractVector
    pz = @. sqrt((1. + δ)^2 - px^2 - py^2)

    newx = @. x + len * px / pz
    newy = @. y + len * py / pz
    newσ = @. σ + len * (1. - β0β * (1. + δ) / pz)

    return newx, px, newy, py, newσ, δ, β0β
end

function ChainRulesCore.rrule(::typeof(driftExact),
        x::T, px::T, y::T, py::T, σ::T, δ::T, β0β::T, len::AbstractFloat) where T<:AbstractArray
    p = driftExact(x, px, y, py, σ, δ, β0β, len)

    function driftExact_pullback(Δ)
        Δx, Δpx, Δy, Δpy, Δσ, Δδ, Δβ0β = Δ
        
        pz = @. sqrt((1. + δ)^2 - px^2 - py^2)
        lOpz2 = @. len / pz^2
        
        newΔpx = @. Δpx + lOpz2 * ((pz + px^2 /pz) * Δx + (py * Δy .- β0β*(1. +δ) * Δσ) * px/pz)
        newΔpy = @. Δpy + lOpz2 * ((pz + py^2 /pz) * Δy + (px * Δx - β0β*(1. +δ) * Δσ) * py/pz)
        
        pOpz = @. (1. + δ) / pz        
        newΔδ = @. Δδ - lOpz2 * (pOpz * (px * Δx + py * Δy) + β0β * (pz - (1. + δ) * pOpz) * Δσ)
        newΔβ0β = @. Δβ0β - len * pOpz * Δσ

        return ChainRulesCore.NoTangent(), Δx, newΔpx, Δy, newΔpy, Δσ, newΔδ, newΔβ0β, ChainRulesCore.NoTangent()
    end
    return p, driftExact_pullback
end

"""
    dipoleEdge(p::T, curvature, edgeAngle, hgap, fint)

Kick particles due to dipole edge effects.
"""
function dipoleEdge(x::T, px::T, y::T, py::T, σ::T, δ::T, β0β::T,
        len::AbstractFloat, α::AbstractFloat, ϵ::AbstractFloat,
        hgap::AbstractFloat, fint::AbstractFloat) where T<:AbstractVector
    corr = 2 * α/len * hgap * fint

    newpx = @. px + x * α/len * tan(ϵ)
    newpy = @. py - y * α/len * tan(ϵ - corr / cos(ϵ) * (1 + sin(ϵ)^2))
    return x, newpx, y, newpy, σ, δ, β0β
end

function ChainRulesCore.rrule(::typeof(dipoleEdge), x::T, px::T, y::T, py::T, σ::T, δ::T, β0β::T,
        len::AbstractFloat, α::AbstractFloat, ϵ::AbstractFloat,
        hgap::AbstractFloat, fint::AbstractFloat) where T<:AbstractVector
    pnew = dipoleEdge(x, px, y, py, σ, δ, β0β, len, α, ϵ, hgap, fint)

    function dipoleEdge_pullback(Δ)
        Δx, Δpx, Δy, Δpy, Δσ, Δδ, Δβ0β = Δ
        corr = 2 * α/len * hgap * fint

        newΔx = @. Δx + α/len * tan(ϵ) * Δpx
        newΔy = @. Δy - α/len * tan(ϵ - corr / cos(ϵ) * (1. + sin(ϵ)^2)) * Δpy

        return ChainRulesCore.NoTangent(), newΔx, Δpx, newΔy, Δpy, Δσ, Δδ, Δβ0β, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
    end

    return pnew, dipoleEdge_pullback
end

function dipoleEdge(x::T, px::T, y::T, py::T, σ::T, δ::T, β0β::T,
        len::AbstractFloat, α::AbstractFloat, ϵ::AbstractFloat) where T<:AbstractVector

    newpx = @. px + x * α/len * tan(ϵ)
    newpy = @. py - y * α/len * tan(ϵ)

    return x, newpx, y, newpy, σ, δ, β0β
end

function ChainRulesCore.rrule(::typeof(dipoleEdge), x::T, px::T, y::T, py::T, σ::T, δ::T, β0β::T,
        len::AbstractFloat, α::AbstractFloat, ϵ::AbstractFloat) where T<:AbstractVector
    pnew = dipoleEdge(x, px, y, py, σ, δ, β0β, len, α, ϵ)

    function dipoleEdge_pullback(Δ)
        Δx, Δpx, Δy, Δpy, Δσ, Δδ, Δβ0β = Δ
        
        newΔx = @. Δx + α/len * tan(ϵ) * Δpx
        newΔy = @. Δy - α/len * tan(ϵ) * Δpy
        
        return ChainRulesCore.NoTangent(), newΔx, Δpx, newΔy, Δpy, Δσ, Δδ, Δβ0β, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
    end

    return pnew, dipoleEdge_pullback
end

"""
    thinMultipole(p::T, length::Float64, kn::Vector{Float64}, ks::Vector{Float64}) where {T<:AbstractArray{Float64}}

Update particle momentum in transversal magnetic field up to arbitrary order.
"""
function thinMultipole(x::T, px::T, y::T, py::T, σ::T, δ::T, β0β::T,
        len::AbstractFloat, kn::S, ks::S) where {S<:AbstractVector,T<:AbstractVector}
    dpx = kn[end]
    dpy = ks[end]

    # assumes kn, ks are of same length
    for i = length(kn)-1:-1:1
        zre = @. (dpx * x - dpy * y) / i
        zim = @. (dpx * y + dpy * x) / i
        dpx = @. kn[i] + zre
        dpy = @. ks[i] + zim
    end

    # update
    newpx = @. px - len * dpx
    newpy = @. py + len * dpy

    return x, newpx, y, newpy, σ, δ, β0β
end

function ChainRulesCore.rrule(::typeof(thinMultipole),
        x::T, px::T, y::T, py::T, σ::T, δ::T, β0β::T,
        len::AbstractFloat, kn::S, ks::S) where {S<:AbstractVector,T<:AbstractVector}
    pnew = thinMultipole(x, px, y, py, σ, δ, β0β, len, kn, ks)

    function thinMultipole_pullback(Δ)
        Δx, Δpx, Δy, Δpy, Δσ, Δδ, Δβ0β = Δ
        # dpx/dx and dpy/dx
        δpx_δx = kn[end]
        δpy_δx = ks[end]
        for i = length(kn)-1:-1:2
            zre = @. (δpx_δx * x - δpy_δx * y) / (i-1)
            zim = @. (δpx_δx * y + δpy_δx * x) / (i-1)
            δpx_δx = zre .+ kn[i]
            δpy_δx = zim .+ ks[i]
        end

        # dpx/dy and dpy/dy
        δpx_δy = -ks[end]
        δpy_δy = kn[end]
        for i = length(kn)-1:-1:2
            zre = @. (δpx_δy * x - δpy_δy * y) / (i-1)
            zim = @. (δpx_δy * y + δpy_δy * x) / (i-1)
            δpx_δy = zre .- ks[i]
            δpy_δy = zim .+ kn[i]
        end

        newΔx = @. Δx - len * (δpx_δx * Δpx + δpx_δy * Δpy)
        newΔy = @. Δy + len * (δpy_δx * Δpx + δpy_δy * Δpy)

        # pullback for normal multipole strengths
        Δkn = zeros(length(kn), length(Δpx))
        Δkn[1,:] = -len .* Δpx

        δpx_δkn = 1
        δpy_δkn = 0
        for i in 2:1:length(kn)
            zre = @. (δpx_δkn * x - δpy_δkn * y) / (i-1)
            zim = @. (δpx_δkn * y + δpy_δkn * x) / (i-1)
            δpx_δkn = zre
            δpy_δkn = zim
            Δkn[i,:] = @. len * (-δpx_δkn * Δpx + δpy_δkn * Δpy)
        end

        # pullback for skew multipole strengths
        Δks = zeros(length(ks), length(Δpy))
        Δks[1,:] = len .* Δpy

        δpx_δks = 0
        δpy_δks = -1
        for i = 2:1:length(ks)
            zre = @. (δpx_δks * x - δpy_δks * y) / (i-1)
            zim = @. (δpx_δks * y + δpy_δks * x) / (i-1)
            δpx_δks = zre
            δpy_δks = zim
            Δks[i,:] = @. len * (δpx_δks * Δpx - δpy_δks * Δpy)
        end

        return ChainRulesCore.NoTangent(), newΔx, Δpx, newΔy, Δpy, Δσ, Δδ, Δβ0β, ChainRulesCore.NoTangent(), sum(Δkn, dims=2), sum(Δks, dims=2)
    end

    return pnew, thinMultipole_pullback
end

function thinMultipole(x::T, px::T, y::T, py::T, σ::T, δ::T, β0β::T,
        len::AbstractFloat, kn::S, ks::S, hx::AbstractFloat, hy::AbstractFloat) where {S<:AbstractVector,T<:AbstractVector}
    dpx = kn[end]
    dpy = ks[end]

    # assumes kn, ks are of same length
    for i = length(kn)-1:-1:1
        zre = @. (dpx * x - dpy * y) / i
        zim = @. (dpx * y + dpy * x) / i
        dpx = @. kn[i] + zre
        dpy = @. ks[i] + zim
    end

    # curvature effects
    hxlx = @. hx * len * x
    hyly = @. hy * len * y

    if len != 0.
        hxx = hxlx ./ len
        hyy = hyly ./ len
    else  # non physical weak focusing disabled (SixTrack mode)
        hxx = 0.
        hyy = 0.
    end

    dpx = @. len * (-1. * dpx + hx + hx * δ - kn[1,:] * hxx)
    dpy = @. len * (dpy - hy - hy * δ + ks[1,:] * hyy)

    dσ = @. (hyly - hxlx) * β0β

    # update
    newpx = @. px + dpx
    newpy = @. py + dpy
    newσ = @. σ + dσ

    return x, newpx, y, newpy, newσ, δ, β0β
end

function ChainRulesCore.rrule(::typeof(thinMultipole), x::T, px::T, y::T, py::T, σ::T, δ::T, β0β::T,
        len::AbstractFloat, kn::S, ks::S, hx::AbstractFloat, hy::AbstractFloat) where {S<:AbstractVector,T<:AbstractVector}
    pnew = thinMultipole(x, px, y, py, σ, δ, β0β, len, kn, ks, hx, hy)

    function thinMultipole_pullback(Δ)
        Δx, Δpx, Δy, Δpy, Δσ, Δδ, Δβ0β = Δ
        
        # dpx/dx and dpy/dx
        δpx_δx = kn[end]
        δpy_δx = ks[end]
        for i = length(kn)-1:-1:2
            zre = @. (δpx_δx * x - δpy_δx * y) / (i-1)
            zim = @. (δpx_δx * y + δpy_δx * x) / (i-1)
            δpx_δx = zre .+ kn[i]
            δpy_δx = zim .+ ks[i]
        end

        # dpx/dy and dpy/dy
        δpx_δy = -ks[end]
        δpy_δy = kn[end]
        for i = length(kn)-1:-1:2
            zre = @. (δpx_δy * x - δpy_δy * y) / (i-1)
            zim = @. (δpx_δy * y + δpy_δy * x) / (i-1)
            δpx_δy = zre .- ks[i]
            δpy_δy = zim .+ kn[i]
        end

        newΔx = @. Δx - len * (δpx_δx * Δpx + δpx_δy * Δpy)
        newΔy = @. Δy + len * (δpy_δx * Δpx + δpy_δy * Δpy)

        # pullback for normal multipole strengths
        Δkn = zeros(length(kn), length(Δpx))
        Δkn[1,:] = -len .* Δpx

        δpx_δkn = 1
        δpy_δkn = 0
        for i in 2:1:length(kn)
            zre = @. (δpx_δkn * x - δpy_δkn * y) / (i-1)
            zim = @. (δpx_δkn * y + δpy_δkn * x) / (i-1)
            δpx_δkn = zre
            δpy_δkn = zim
            Δkn[i,:] = @. len * (-δpx_δkn * Δpx + δpy_δkn * Δpy)
        end

        # pullback for skew multipole strengths
        Δks = zeros(length(ks), length(Δpy))
        Δks[1,:] = len .* Δpy

        δpx_δks = 0
        δpy_δks = -1
        for i = 2:1:length(ks)
            zre = @. (δpx_δks * x - δpy_δks * y) / (i-1)
            zim = @. (δpx_δks * y + δpy_δks * x) / (i-1)
            δpx_δks = zre
            δpy_δks = zim
            Δks[i,:] = @. len * (δpx_δks * Δpx - δpy_δks * Δpy)
        end


        # curvature effects        
        newΔx .-= kn[1,:].*hx.*len .* Δpx .+ hx.*len.*β0β .* Δσ
        newΔpx = Δpx .+ ks[2,:].*hy.*len .* Δpy .+ hy.*len.*β0β .* Δσ
        newΔδ = Δδ .+ hx.*len .* Δpx .- hy.*len * Δpy
        newΔβ0β = Δβ0β .+ (hy.*len .* y .- hx.*len .* x) .* Δσ
        
        Δkn[1,:] .-= hx.*len .* x .* Δpx
        Δks[1,:] .+= hy.*len .* y .* Δpy
        
        return ChainRulesCore.NoTangent(), newΔx, newΔpx, newΔy, Δpy, Δσ, newΔδ, newΔβ0β, ChainRulesCore.NoTangent(), sum(Δkn, dims=2), sum(Δks, dims=2), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
    end

    return pnew, thinMultipole_pullback
end