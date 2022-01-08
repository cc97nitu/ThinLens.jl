"""Container for beamline elements."""
abstract type BeamlineElement end


"""Drift."""
mutable struct Drift <: BeamlineElement
    len::Float64
end

function (e::Drift)(p::T) where {T<:AbstractArray{Float64}}
    driftExact(p, e.len)
end


"""Element that can be described via transversal magnetic fields."""
abstract type Magnet <: BeamlineElement end

function (e::Magnet)(p::T) where {T<:AbstractArray{Float64}}
    for (c, d) in zip(e.splitScheme.c, e.splitScheme.d)
        # drift
        p = driftExact(p, c * e.len)
    
        # kick
        if d != 0
            p = thinMultipole(p, d * e.len, e.kn, e.ks)
        end
    end

    return p
end

"""
    setMultipole!(magnet::Magnet, order::Int, kn::Number, ks::Number)

Set normal and skew multipole values for given order.

# Arguments
- 'order::Int': order of multipole, e.g. 1st for quadrupoles, 3rd for octupoles.
"""
function setMultipole!(magnet::Magnet, order::Int, kn::Number, ks::Number)
    if length(magnet.kn) < order + 1
        # need to extend multipole array
        newKn = zeros(order + 1); newKs = zeros(order + 1)
        
        for (i, kn) in enumerate(magnet.kn)
            newKn[i] = kn
        end

        for (i, ks) in enumerate(magnet.ks)
            newKs[i] = ks
        end

        magnet.kn = newKn
        magnet.ks = newKs
    end

    magnet.kn[order + 1] = kn
    magnet.ks[order + 1] = ks
end

"""Quadrupole."""
mutable struct Quadrupole <: Magnet
    len::Float64
    kn::Vector{Float64}
    ks::Vector{Float64}
    splitScheme::SplitScheme
end

Quadrupole(len::Number, k1n::Number, k1s::Number; split=splitO2nd) = Quadrupole(len, [0., k1n, 0.], [0., k1s, 0.], split)

"""Sextupole."""
mutable struct Sextupole <: Magnet
    len::Float64
    kn::Vector{Float64}
    ks::Vector{Float64}
    splitScheme::SplitScheme
end

Sextupole(len::Number, k2n::Number, k2s::Number; split=splitO2nd) = Sextupole(len, [0., 0., k2n], [0., 0., k2s], split)

"""Bending magnet."""
mutable struct BendingMagnet <: Magnet
    len::Float64
    kn::Vector{Float64}
    ks::Vector{Float64}
    α::Float64  # horizontal deflection angle ref. trajectory
    β::Float64  # vertical deflection angle ref. trajectory
    ϵ1::Float64
    ϵ2::Float64
    splitScheme::SplitScheme
end

SBen(len::Number, α::Number, ϵ1::Number, ϵ2::Number, splitScheme::SplitScheme=splitO2nd) = BendingMagnet(len, [α / len, 0., 0.], [0., 0., 0.], α, 0., ϵ1, ϵ2, splitScheme)

RBen(len::Number, α::Number, ϵ1::Number, ϵ2::Number, splitScheme::SplitScheme=splitO2nd) = BendingMagnet(len, [α / len, 0., 0.], [0., 0., 0.], α, 0., ϵ1 + α/2, ϵ2 + α/2, splitScheme)

function (e::BendingMagnet)(p::T) where {T<:AbstractArray{Float64}}
    # entry pole face
    p = dipoleEdge(p, e.len, e.α, e.ϵ1)

    for (c, d) in zip(e.splitScheme.c, e.splitScheme.d)
        # drift
        p = driftExact(p, c * e.len)
    
        # kick
        if d != 0
            # p = thinMultipole(p, d * e.len, e.kn, e.ks)
            # p = curvatureEffectKick(p, d * e.len, e.kn, e.ks, e.α/e.len, e.β/e.len)

            p = thinMultipole(p, d * e.len, e.kn, e.ks, e.α/e.len, e.β/e.len)

        end
    end

    # exit pole face
    p = dipoleEdge(p, e.len, e.α, e.ϵ2)
    
    return p
end