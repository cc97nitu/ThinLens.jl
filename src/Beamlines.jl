
import Flux

function FlatChain(cells, args...; kwargs...)
    beamline = []
    for cell in cells
        for element in cell
            push!(beamline, element)
        end
    end
    
    return Flux.Chain(beamline..., args...; kwargs...)
end

function NestedChain(cells, args...; kwargs...)
    beamline = []
    for cell in cells
        push!(beamline, cell)
    end
    
    return Flux.Chain(beamline..., args...; kwargs...)
end

"""
    track(model::Flux.Chain, batch::DenseArray)::PermutedDimsArray

Track particles through model and collect phasespace after each layer.
"""
function track(model::Flux.Chain, batch::DenseArray)::PermutedDimsArray
    out = reduce(hcat, Flux.activations(model, batch))
    out = reshape(out, 7, :, length(model))  # dim, particle, BPM
    return PermutedDimsArray(out, (1,3,2))  # dim, BPM, particle
end

precompile(track, (Flux.Chain{NTuple{12, Flux.Chain{Tuple{Drift, BendingMagnet, Drift, BendingMagnet, Drift, Sextupole, Drift, Quadrupole, Drift, Quadrupole, Drift, Sextupole, Drift, Drift, Drift}}}}, Vector{Float64}))
precompile(track, (Flux.Chain{NTuple{12, Flux.Chain{Tuple{Drift, BendingMagnet, Drift, BendingMagnet, Drift, Sextupole, Drift, Quadrupole, Drift, Quadrupole, Drift, Sextupole, Drift, Drift, Drift}}}}, Matrix{Float64}))

"""
    assignMasks(model; nested=true, quadMask=nothing, sextMask=nothing)::IdDict{Vector{Float64}, Vector{Float64}}

Return dict which maps kn to their corresponding mask.
"""
function assignMasks(model::Flux.Chain;
    nested::Bool=true, quadMask::Union{Vector{Float64}, Nothing}=nothing, sextMask::Union{Vector{Float64}, Nothing}=nothing
    )::IdDict{Vector{Float64}, Vector{Float64}}

    masks = IdDict{Vector{Float64}, Vector{Float64}}()

    if nested
        for cell in model
            for element in cell
                if typeof(element) == Quadrupole && !isnothing(quadMask) 
                    masks[element.kn] = quadMask
                elseif typeof(element) == Sextupole && !isnothing(sextMask)
                    masks[element.kn] = sextMask
                end
            end
        end
    else
        for element in model
            if typeof(element) == Quadrupole && !isnothing(quadMask) 
                masks[element.kn] = quadMask
            elseif typeof(element) == Sextupole && !isnothing(sextMask)
                masks[element.kn] = sextMask
            end
        end
    end

    return masks
end