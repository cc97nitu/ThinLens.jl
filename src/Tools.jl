module Tools

using Statistics
import Flux
import FFTW
import EasyFit
import ThinLens as TL

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

precompile(getTuneChroma_fft, (Flux.Chain{NTuple{12, Flux.Chain{Tuple{TL.Drift, TL.BendingMagnet, TL.Drift, TL.BendingMagnet, TL.Drift, TL.Sextupole, TL.Drift, TL.Quadrupole, TL.Drift, TL.Quadrupole, TL.Drift, TL.Sextupole, TL.Drift, TL.Drift, TL.Drift}}}},
TL.Beam, Int))

end  # module Tools