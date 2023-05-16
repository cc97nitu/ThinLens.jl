module Tools

using Statistics
import LinearAlgebra as LA
import Flux
import FFTW
import EasyFit
import ThinLens as TL


include("Matching.jl")


"""
    getTuneChroma_fft(model, beam::TL.Beam, turns::Int)

Perform FFT of particle positions at the end of the beamline and obtain tune / chromaticities from linear fit.
"""
function getTuneChroma_fft(model, beam::TL.Beam, turns::Int)
    # create particles
    particles = zeros(7, 5)
    particles[1,:] .= 1e-6
    particles[3,:] .= 1e-6
    particles[6,:] .= [-5e-3, -5e-4, 0., 5e-4, 5e-3]
    
    # adapt velocity ratio β0/β
    p = beam.p .* (1. .+ particles[6,:])  # GeV/c
    E = sqrt.(p.^2 .+ beam.mass^2)  # GeV
    γ = E ./ beam.mass
    β = p ./ (γ .* beam.mass)
    particles[7,:] .= beam.beta ./ β
    
    # observe horizontal / vertical motion
    longChain = Flux.Chain([model for _ in 1:turns]...)
    track = TL.track(longChain, particles)[[1,3],:,:]  # x,y only
    
    # observe maximum frequency (=tune)
    freqs = FFTW.rfftfreq(turns)
    fft = FFTW.rfft(track .- mean(track, dims=2), 2)  # remove dispersion
    
    xTunes = [freqs[argmax(abs.(fft[1,:,i]))] for i in 1:size(particles)[2]]
    yTunes = [freqs[argmax(abs.(fft[2,:,i]))] for i in 1:size(particles)[2]]
    
    # fit tune and chroma
    xFit = EasyFit.fitlinear(particles[6,:], xTunes)
    yFit = EasyFit.fitlinear(particles[6,:], yTunes)

    return (xFit.b, yFit.b), (xFit.a, yFit.a)
end

"""
    getTuneChroma_nested(model, beam::TL.Beam, turns::Int)

Perform FFT of particle positions to obtain tune / chromaticities from linear fit.
This function assumes a nested model with equally spaced outputs.
"""
function getTuneChroma_nested(model, beam::TL.Beam, turns::Int)
    # create particles
    particles = zeros(7, 5)
    particles[1,:] .= 1e-6
    particles[3,:] .= 1e-6
    particles[6,:] .= [-5e-3, -5e-4, 0., 5e-4, 5e-3]
    
    # adapt velocity ratio β0/β
    p = beam.p .* (1. .+ particles[6,:])  # GeV/c
    E = sqrt.(p.^2 .+ beam.mass^2)  # GeV
    γ = E ./ beam.mass
    β = p ./ (γ .* beam.mass)
    particles[7,:] .= beam.beta ./ β
    
    # observe horizontal / vertical motion
    pos = Array{Float64}(undef, 2, turns*length(model), size(particles, 2))
    
    coords = particles
    for turn in 0:turns-1
        global track = TL.track(model, coords)

        idx = turn*length(model)+1
        pos[:,idx:idx+length(model)-1,:] = track[[1,3],:,:]  # x,y only
        
        coords = track[:,end,:]
    end
    
    # observe maximum frequency (=tune)
    freqs = FFTW.rfftfreq(turns*length(model), length(model))
    fft = FFTW.rfft(pos .- Statistics.mean(pos, dims=2), 2)  # remove dispersion
    
    xTunes = [freqs[argmax(abs.(fft[1,:,i]))] for i in 1:size(particles)[2]]
    yTunes = [freqs[argmax(abs.(fft[2,:,i]))] for i in 1:size(particles)[2]]
    
    # fit tune and chroma
    xFit = EasyFit.fitlinear(particles[6,:], xTunes)
    yFit = EasyFit.fitlinear(particles[6,:], yTunes)

    return (xFit.b, yFit.b), (xFit.a, yFit.a)
end

precompile(getTuneChroma_fft, (Flux.Chain{NTuple{12, Flux.Chain{Tuple{TL.Drift, TL.BendingMagnet, TL.Drift, TL.BendingMagnet, TL.Drift, TL.Sextupole, TL.Drift, TL.Quadrupole, TL.Drift, TL.Quadrupole, TL.Drift, TL.Sextupole, TL.Drift, TL.Drift, TL.Drift}}}},
TL.Beam, Int))

"""
    getTunes_jacobian(model)

Calculate tunes of nested model by linearizing the one-turn map with reverse-mode AD.
"""
function getTunes_jacobian(model)
    origin = zeros(7)

    μ = zeros(2)
    for cell in model
        jacOT = Flux.jacobian(cell, origin)[1]
    
        # get phase advance
        trMX = jacOT[1,1] + jacOT[2,2]
        trMY = jacOT[3,3] + jacOT[4,4]

        μ .+= [acos(1/2 * trMX), acos(1/2 * trMY)]
    end

    return μ ./ (2*π)
end

"""
    function twiss(model)::Dict{Symbol, Any}

Calculate tunes, phase advances and twiss parameters in case no coupling is present.
"""
function twiss(model)::Dict{Symbol, Any}
    # linearize transport maps 
    jacs = []; elementCount::Int = 0

    for cell in model
        for element in cell
            push!(jacs, Flux.jacobian(element, [0.,0.,0.,0.,0.,0.,1.])[1])
            elementCount += 1
        end
    end

    # linear one-turn map
    jac::Matrix{Float64} = Matrix(1.0LA.I, 7, 7)
    for i in jacs
        jac = i * jac
    end

    # is coupling present?
    if LA.norm(jac[1:2,3:4]) + LA.norm(jac[3:4,1:2]) != 0
        println("coupling present")
        throw(ArgumentError("coupling present"))
    end

    # does a periodic solution exist?
    cosXμ = 1/2 * LA.tr(jac[1:2,1:2])
    if abs(cosXμ) > 1
        throw(DomainError(cosXμ, "horizontal cosine(phaseAdvance) out of bounds"))
    end

    cosYμ = 1/2 * LA.tr(jac[3:4,3:4])
    if abs(cosYμ) > 1
        throw(DomainError(cosYμ, "vertical cosine(phaseAdvance) out of bounds"))
    end

    # initial twiss values from one-turn map
    βX = Array{Float64}(undef, elementCount+1); αX = Array{Float64}(undef, elementCount+1)
    sinXμ = sign(jac[1,2]) * sqrt(1 - cosXμ^2)
    βX[1] = jac[1,2] / sinXμ
    αX[1] = 1/(2 * sinXμ) * (jac[1,1] - jac[2,2])

    βY = Array{Float64}(undef, elementCount+1); αY = Array{Float64}(undef, elementCount+1)
    sinYμ = sign(jac[3,4]) * sqrt(1 - cosYμ^2)
    βY[1] = jac[3,4] / sinYμ
    αY[1] = 1/(2 * sinYμ) * (jac[3,3] - jac[4,4])
    
    # propagate twiss along the beamline
    twissTransportX = Array{Float64}(undef, 3, 3, elementCount)
    for (i, m) in enumerate(jacs)
        c = m[1,1]; cp = m[2,1]; s = m[1,2]; sp = m[2,2]
        twissTransportX[:,:,i] = [c^2 -2*s*c s^2; -1*c*cp s*cp+sp*c -1*s*sp; cp^2 -2*sp*cp sp^2]
    end

    twiss = [βX[1], αX[1], (1 + αX[1]^2)/βX[1]]
    for i in 1:elementCount
        twiss = twissTransportX[:,:,i] * twiss
        βX[i+1] = twiss[1]
        αX[i+1] = twiss[2]
    end

    twissTransportY = Array{Float64}(undef, 3, 3, elementCount)
    for (i, m) in enumerate(jacs)
        c = m[3,3]; cp = m[4,3]; s = m[3,4]; sp = m[4,4]
        twissTransportY[:,:,i] = [c^2 -2*s*c s^2; -1*c*cp s*cp+sp*c -1*s*sp; cp^2 -2*sp*cp sp^2]
    end

    twiss = [βY[1], αY[1], (1 + αY[1]^2)/βY[1]]
    for i in 1:elementCount
        twiss = twissTransportY[:,:,i] * twiss
        βY[i+1] = twiss[1]
        αY[i+1] = twiss[2]
    end
    
    # calculate phase advance
    μX = Array{Float64}(undef, elementCount)
    for i in 1:elementCount
        μX[i] = asin(jacs[i][1,2] / sqrt(βX[i] * βX[i+1]))
    end

    μY = Array{Float64}(undef, elementCount)
    for i in 1:elementCount
        μY[i] = asin(jacs[i][3,4] / sqrt(βY[i] * βY[i+1]))
    end
    
    return Dict(:q1=>sum(μX) / (2*π), :q2=>sum(μY) / (2*π), :βX=>βX, :αX=>αX, :βY=>βY, :αY=>αY, :μX=>μX, :μY=>μY)
end

""""
    ellipse(x::Real, px::Real, a::Real, b::Real, θ::Real)

Compute x,y points of ellipse with semi-axes a,b centered around (x, px).
"""
function ellipse(x::Real, px::Real, a::Real, b::Real, θ::Real)
    ψ = LinRange(0, 2π, 500)
    ar = a .* sin.(ψ)
    br = b .* cos.(ψ)
    return x .+ cos(θ) .* ar .- sin(θ) .* br, px .+ sin(θ) .* ar .+ cos(θ) .* br
end 

"""
    twissEllipse(β::Real, α::Real, ϵ::Real; x0=0., px0=0.)

Compute x,y points of twiss ellipse centered around (x0, px0).
"""
function twissEllipse(β::Real, α::Real, ϵ::Real; x0=0., px0=0.)
    γ = (1+α^2)/β
    beamMatrix = ϵ * [β -α; -α γ]

    eigenval, eigenvectors = LA.eigen(beamMatrix)
    a = sqrt(eigenval[1])
    b = sqrt(eigenval[2])
    rotAngle = atan(eigenvectors[2,1], eigenvectors[1,1])

    return ellipse(x0, px0, a, b, rotAngle)
end

"""
    twissEllipse(σ::AbstractMatrix; x0=0., px0=0.)

Compute x,y points of twiss ellipse centered around (x0, px0) from beam matrix σ.
"""
function twissEllipse(σ::AbstractMatrix; x0=0., px0=0.)
    eigenval, eigenvectors = LA.eigen(σ)
    a = sqrt(eigenval[1])
    b = sqrt(eigenval[2])
    rotAngle = atan(eigenvectors[2,1], eigenvectors[1,1])

    return ellipse(x0, px0, a, b, rotAngle)
end

end  # module Tools