import Distributions

"""
A particle beam similar to MAD-X beam.
"""
struct Beam
    E::Float64  # energy in GeV / c
    p::Float64  # momentum in GeV / c
    mass::Float64  # mass in GeV / c^2
    charge::Float64  # charge in e
    ϵx::Float64  # horizontal geometric emittance in m
    ϵy::Float64  # vertical geometric emittance in m
    sigt::Float64  # longitudianl standard deviation in m
    sige::Float64  # ΔE/E standard deviation in 1
    gamma::Float64
    beta::Float64
    centroid::Vector{Float64}
end

"""
    Beam(;energy::Number, mass::Number, charge::Number, ϵx::Number, ϵy::Number, ϵNormed::bool=true, sigt::Float64=0., sige::Float64=0., centroid::Vector{Float64}=zeros(6))

Create a beam.
"""
function Beam(;energy::Number, mass::Number, charge::Number, ϵx::Number, ϵy::Number, ϵNormed::Bool=true, sigt::Float64=0., sige::Float64=0., centroid::Vector{Float64}=zeros(6))
    γ = energy / mass
    momentum = sqrt(energy^2 - mass^2)  # GeV/c
    β = momentum / (γ * mass)
    
    if ϵNormed
        # convert to geometric emittance
        ϵx /= (β * γ)
        ϵy /= (β * γ)
    end

    Beam(energy, momentum, mass, charge, ϵx, ϵy, sigt, sige, γ, β, centroid)
end

"""
    setVelocityRatio!(particles::DenseArray{Float64}, beam::Beam)

Adapt velocity ratio β0/β according to beam properties.
"""
function setVelocityRatio!(particles::DenseArray{Float64}, beam::Beam)
    p = beam.p .* (1. .+ particles[6,:])  # GeV/c
    E = sqrt.(p.^2 .+ beam.mass^2)  # GeV
    γ = E ./ beam.mass
    β = p ./ (γ .* beam.mass)
    particles[7,:] .= beam.beta ./ β
end

"""
    ParticlesGaussian(beam::Beam, size::Integer; cutoff::Number=3., twiss::Vector{Float64}=[1., 0., 1., 0.])

Sample particles for beam from (transversally) truncated normal distribution.
"""
function ParticlesGaussian(beam::Beam, size::Integer; cutoff::Number=3., twiss::Vector{Float64}=[1., 0., 1., 0.])
    particles = Array{Float64}(undef, 7, size)

    (βx, αx, βy, αy) = twiss
    
    # sample transverse coordinates
    angleDist = Distributions.Uniform(0, 2π)
    horActionDist = Distributions.TruncatedNormal(0, beam.ϵx / 2., 0, cutoff*beam.ϵx / 2.)
    verActionDist = Distributions.TruncatedNormal(0, beam.ϵy / 2., 0, cutoff*beam.ϵy / 2.)

    horPhase = Array{Float64}(undef, size); horAction = Array{Float64}(undef, size)
    Distributions.rand!(angleDist, horPhase)
    Distributions.rand!(horActionDist, horAction)
    
    particles[1,:] .= beam.centroid[1] .+ sqrt.(2 * βx .* horAction) .* cos.(horPhase)
    particles[2,:] .= beam.centroid[2] .- sqrt.(2 .* horAction ./ βx) .* (sin.(horPhase) .+ αx .* cos.(horPhase))
    
    verPhase = Array{Float64}(undef, size); verAction = Array{Float64}(undef, size)
    Distributions.rand!(angleDist, verPhase)
    Distributions.rand!(verActionDist, verAction)
    
    particles[3,:] .= beam.centroid[3] .+ sqrt.(2 * βy .* verAction) .* cos.(verPhase)
    particles[4,:] .= beam.centroid[4] .- sqrt.(2 .* verAction ./ βy) .* (sin.(verPhase) .+ αy .* cos.(verPhase))
    
    # sample longitudinal coordinates
    if beam.sigt == 0.
        particles[5,:] = zeros(size)
    else
        longDist = Distributions.TruncatedNormal(0., beam.sigt, -1*cutoff*beam.sigt, cutoff*beam.sigt)
        long = Array{Float64}(undef, size)
        Distributions.rand!(longDist, long)
        particles[5,:] .= long
    end
    particles[5,:] .+= beam.centroid[5]

    if beam.sige == 0.
        particles[6,:] = zeros(size)
    else
        δDist = Distributions.TruncatedNormal(0., beam.sige*beam.beta, -1*cutoff*beam.sige*beam.beta, cutoff*beam.sige*beam.beta)
        δ = Array{Float64}(undef, size)
        Distributions.rand!(δDist, δ)
        particles[6,:] .= δ
    end
    particles[6,:] .+= beam.centroid[6]
    
    # adapt velocity ratio β0/β
    setVelocityRatio!(particles, beam)
    
    return particles
end

