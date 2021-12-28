
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