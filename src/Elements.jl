"""Container for beamline elements."""
abstract type BeamlineElement end


"""Drift."""
mutable struct Drift <: BeamlineElement
    len::Float64
end

function (e::Drift)(particles::AbstractVecOrMat)
    p = [particles[i,:] for i in 1:size(particles,1)]

    p = driftExact(p..., e.len)
    return vcat(transpose.(p)...)
end


"""Element that can be described via transversal magnetic fields."""
abstract type Magnet <: BeamlineElement end

function (e::Magnet)(particles::AbstractVecOrMat)
    p = [particles[i,:] for i in 1:size(particles,1)]
    
    stepLength = e.len / e.steps
    for _ in 1:e.steps        
        for (c, d) in zip(e.splitScheme.c, e.splitScheme.d)
            # drift
            p = driftExact(p..., c * stepLength)
        
            # kick
            if d != 0
                p = thinMultipole(p..., d * stepLength, e.kn, e.ks)
            end
        end
    end

    return vcat(transpose.(p)...)
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
    steps::Int
end

Quadrupole(len::Number, k1n::Number, k1s::Number; split::SplitScheme=splitO2nd, steps::Int=1) = Quadrupole(len, [0., k1n, 0., 0.], [0., k1s, 0., 0.], split, steps)

"""Sextupole."""
mutable struct Sextupole <: Magnet
    len::Float64
    kn::Vector{Float64}
    ks::Vector{Float64}
    splitScheme::SplitScheme
    steps::Int
end

Sextupole(len::Number, k2n::Number, k2s::Number; split::SplitScheme=splitO2nd, steps::Int=1) = Sextupole(len, [0., 0., k2n, 0.], [0., 0., k2s, 0.], split, steps)

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
    steps::Int
end

SBen(len::Number, α::Number, ϵ1::Number, ϵ2::Number; split::SplitScheme=splitO2nd, steps::Int=1) = BendingMagnet(len, [α / len, 0., 0., 0.], [0., 0., 0., 0.], α, 0., ϵ1, ϵ2, split, steps)

RBen(len::Number, α::Number, ϵ1::Number, ϵ2::Number; split::SplitScheme=splitO2nd, steps::Int=1) = BendingMagnet(len, [α / len, 0., 0., 0.], [0., 0., 0., 0.], α, 0., ϵ1 + α/2, ϵ2 + α/2, split, steps)

function (e::BendingMagnet)(particles::AbstractVecOrMat)
    p = [particles[i,:] for i in 1:size(particles,1)]
    
    # entry pole face
    p = dipoleEdge(p..., e.len, e.α, e.ϵ1)

    stepLength = e.len / e.steps
    for _ in 1:e.steps
        for (c, d) in zip(e.splitScheme.c, e.splitScheme.d)
            # drift
            p = driftExact(p..., c * stepLength)
        
            # kick
            if d != 0
                p = thinMultipole(p..., d * stepLength, e.kn, e.ks, e.α/e.len, e.β/e.len)
            end
        end
    end

    # exit pole face
    p = dipoleEdge(p..., e.len, e.α, e.ϵ2)
    
    return vcat(transpose.(p)...)
end