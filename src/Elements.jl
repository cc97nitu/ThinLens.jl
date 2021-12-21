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

"""Quadrupole."""
mutable struct Quadrupole <: Magnet
    len::Float64
    kn::Vector{Float64}
    ks::Vector{Float64}
    splitScheme::SplitScheme
end

Quadrupole(len::Number, k1n::Number, k1s::Number) = Quadrupole(len, [0., k1n, 0.], [0., k1s, 0.], splitO2nd)

"""Sextupole."""
mutable struct Sextupole <: Magnet
    len::Float64
    kn::Vector{Float64}
    ks::Vector{Float64}
    splitScheme::SplitScheme
end

Sextupole(len::Number, k2n::Number, k2s::Number) = Sextupole(len, [0., 0., k2n], [0., 0., k2s], splitO2nd)

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
            p = thinMultipole(p, d * e.len, e.kn, e.ks)
            p = curvatureEffectKick(p, d * e.len, e.kn, e.ks, e.α/e.len, e.β/e.len)
        end
    end

    # exit pole face
    p = dipoleEdge(p, e.len, e.α, e.ϵ2)
    
    return p
end