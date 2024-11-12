
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
function track(model::Flux.Chain, batch::AbstractVecOrMat)::PermutedDimsArray
    out = reduce(hcat, Flux.activations(model, batch))
    out = reshape(out, 7, :, length(model))  # dim, particle, BPM
    return PermutedDimsArray(out, (1,3,2))  # dim, BPM, particle
end

# not differentiable :(
function track(model::Flux.Chain, batch::AbstractVecOrMat, turns::Int)::Array{Float64}
    coordinateBuffer = Array{Float64}(undef, 7, turns*length(model), size(batch)[2])  # dim, BPM, particle

    # first turn
    out = reduce(hcat, Flux.activations(model, batch))
    out = reshape(out, 7, :, length(model))  # dim, particle, BPM
    coordinateBuffer[:, 1:length(model) , :] = PermutedDimsArray(out, (1,3,2))  # dim, BPM, particle

    for turn in 2:turns
        out = reduce(hcat, Flux.activations(model, coordinateBuffer[:,(turn - 1)*length(model),:]))
        out = reshape(out, 7, :, length(model))  # dim, particle, BPM
        coordinateBuffer[:, 1 + (turn - 1)*length(model):turn*length(model), :] = PermutedDimsArray(out, (1,3,2))  # dim, BPM, particle
    end
    return coordinateBuffer
end

precompile(track, (Flux.Chain{NTuple{12, Flux.Chain{Tuple{Drift, BendingMagnet, Drift, BendingMagnet, Drift, Sextupole, Drift, Quadrupole, Drift, Quadrupole, Drift, Sextupole, Drift, Drift, Drift}}}}, Vector{Float64}))
precompile(track, (Flux.Chain{NTuple{12, Flux.Chain{Tuple{Drift, BendingMagnet, Drift, BendingMagnet, Drift, Sextupole, Drift, Quadrupole, Drift, Quadrupole, Drift, Sextupole, Drift, Drift, Drift}}}}, Matrix{Float64}))

"""
    assignMasks(model; nested=true, quadMask=nothing, sextMask=nothing)::IdDict{Vector{Float64}, Vector{Float64}}

Return dict which maps kn to their corresponding mask.
"""
function assignMasks(model::Flux.Chain;
    nested::Bool=true, quadMask::Union{Vector{Float64}, Nothing}=nothing, tripletMask::Union{Vector{Float64}, Nothing}=nothing,
    sextMask::Union{Vector{Float64}, Nothing}=nothing, bendMask::Union{Vector{Float64}, Nothing}=nothing
    )::IdDict{Vector{Float64}, Vector{Float64}}

    masks = IdDict{Vector{Float64}, Vector{Float64}}()

    if nested
        for cell in model
            for element in cell
                if typeof(element) == Quadrupole && !isnothing(quadMask)
                    if element.len == 0.4804
                        masks[element.kn] = tripletMask
                    else
                        masks[element.kn] = quadMask
                    end
                elseif typeof(element) == Sextupole && !isnothing(sextMask)
                    masks[element.kn] = sextMask
                elseif typeof(element) == BendingMagnet && !isnothing(bendMask)
                    masks[element.kn] = bendMask
                end
            end
        end
    else
        for element in model
            if typeof(element) == Quadrupole && !isnothing(quadMask) 
                if element.len == 0.4804
                    masks[element.kn] = tripletMask
                else
                    masks[element.kn] = quadMask
                end
        elseif typeof(element) == Sextupole && !isnothing(sextMask)
                masks[element.kn] = sextMask
            elseif typeof(element) == BendingMagnet && !isnothing(bendMask)
                masks[element.kn] = bendMask
            end
        end
    end

    return masks
end

macro track_oneTurn(noCells)
    results = [Symbol("z", i) for i in 1:noCells]
    
    body = Expr(:block)
    push!(body.args, :($(results[1]) = model[1](particles)))

    for i in 2:length(results)
        push!(body.args, :($(results[i]) = model[$i]( $(results[i-1]) )))
    end
    push!(body.args, :(out = cat($(results...), dims=3)))
    push!(body.args, :(permutedims(out, (1,3,2))))
    
    head = :(track_oneTurn(model::Flux.Chain, particles::AbstractVecOrMat)::AbstractArray)
    return Expr(:function, head, body)
end

macro track(turns)
    results = [Symbol("z", i) for i in 1:turns]
    
    body = Expr(:block)
    push!(body.args, :($(results[1]) = track_oneTurn(model, particles)))

    for i in 2:length(results)
        push!(body.args, :($(results[i]) = track_oneTurn(model, $(results[i-1])[:,end,:])))
    end
    push!(body.args, :(cat($(results...), dims=2)))
    
    head = :(track_long(model::Flux.Chain, particles::AbstractVecOrMat))
    return Expr(:function, head, body)
end

macro track_preTrack(turns)
    results = [Symbol("z", i) for i in 1:turns+1]
    
    body = Expr(:block)
    push!(body.args, :($(results[1]) = reshape( preTrack(particles), (size(particles,1), 1, size(particles,2))) ))

    for i in 2:length(results)
        push!(body.args, :($(results[i]) = track_oneTurn(model, $(results[i-1])[:,end,:])))
    end
    push!(body.args, :(cat($(results...), dims=2)))
    
    head = :(track_long(preTrack::Flux.Chain, model::Flux.Chain, particles::AbstractVecOrMat))
    return Expr(:function, head, body)
end