# module Polynomial

import Flux
import ChainRulesCore
import TaylorSeries as TS

# export PolyN


struct PolyN{S,T}
    coefficients::Vector{Vector{S}}
    coeffTable::Vector{Vector{Vector{T}}}
end


# function PolyN(t::Vector{TS.TaylorN{T}}) where T<:Number
#     coeffTable = TS.coeff_table |> copy
#     noCoeff = coeffTable .|> length |> sum
    
#     coefficients = [Vector{TS.numtype(t[1])}(undef, noCoeff) for _ in eachindex(t)]
    
#     for n in eachindex(coefficients)
#         c = 1
#         for i in eachindex(coeffTable)
#             for j in eachindex(coeffTable[i])
#                 coefficients[n][c] = TS.getcoeff(t[n], coeffTable[i][j])

#                 c += 1
#             end
#         end
#     end
    
#     return PolyN(coefficients, coeffTable)
# end

function PolyN(t::Vector{TS.TaylorN{T}}) where T<:Number
    coefficients = [[TS.getcoeff(t[k], TS.coeff_table[1][1]) |> Float64] for k in eachindex(t)]
    coeffTable = [[TS.coeff_table[1][1]]]

    # iterate over multi-indices
    for i in 2:length(TS.coeff_table)  # iterate over orders
        allidx = []
        for j in eachindex(TS.coeff_table[i])
            allcoeff = [TS.getcoeff(t[k], TS.coeff_table[i][j]) for k in eachindex(t)]
            iszero(allcoeff) && continue
            
            push!(allidx, TS.coeff_table[i][j])
            for k in eachindex(allcoeff)
                push!(coefficients[k], allcoeff[k] .|> Float64)
            end
        end
        
        length(allidx) > 0 && push!(coeffTable, allidx)
    end
    
    coefficients = identity.([identity.(coefficients[i]) for i in eachindex(coefficients)])
    coeffTable = identity.([identity.(coeffTable[k]) for k in eachindex(coeffTable)])
    
    return PolyN(coefficients, coeffTable)
end

function evaluate(p::PolyN{S,T}, z::AbstractVector{U}, ndims::T) where {S<:Number,T<:Int,U<:Number}
    result = Vector{U}(undef, ndims)
    
    @inbounds for n in 1:ndims
        curres = p.coefficients[n][1]

        c = 1
        for i in 2:length(p.coeffTable)
            for j in eachindex(p.coeffTable[i])
                c += 1
                iszero(p.coefficients[n][c]) && continue

                f = p.coefficients[n][c]
                for k in eachindex(p.coeffTable[i][j])
                    f *= z[k] ^ p.coeffTable[i][j][k]
                end

                curres += f
            end
        end
        
        result[n] = curres
    end
    
    return result
end


function (p::PolyN{S,T})(z::AbstractVector{U}) where {S<:Number,T<:Int,U<:Number}
    @assert length(z) == length(p.coeffTable[1][1]) "got input dimension $(length(z)), expected $(length(p.coeffTable[1][1]))"
    ndims = length(p.coefficients)

    return evaluate(p, z, ndims)
end


struct PolyMN{S,T}
    polynomials::Array{PolyN{S,T}, 1}
end


# function PolyMN(m::Matrix{TaylorSeries.TaylorN{S}}) where S<:Number
#     polynomials = Vector{PolyN}(undef, size(m)[1])

#     for row in eachindex(polynomials)
#         polynomials[row] = PolyN(m[row,:])
#     end
    
#     return PolyMN(polynomials)
# end


function PolyMN(m)
    x = PolyN(m[1,:])
    polynomials = Vector{typeof(x)}(undef, size(m)[1])

    for row in eachindex(polynomials)
        polynomials[row] = PolyN(m[row,:])
    end
    
    return PolyMN(polynomials)
end


function (p::PolyMN)(z::AbstractVector{U}) where U<:Number
    result = Matrix{U}(undef, length(p.polynomials), length(p.polynomials[1].coefficients))
    
    Threads.@threads for row in 1:size(result)[1]
        result[row,:] .= p.polynomials[row](z)
    end
    
    return result
end


dummy_PolyN() = PolyN([zeros(1),], [[[0,],],])
dummy_PolyMN() = PolyMN([dummy_PolyN(),])


struct PolyLayer{S,T}
    poly::PolyN{S,T}
    gradient::PolyMN{S,T}
end


function CallPolyLayer(p::PolyLayer, particles::Matrix{T}) where T
    out = copy(particles)
    return p.poly(out)
end


function ChainRulesCore.rrule(::typeof(CallPolyLayer), p::PolyLayer, particles::Matrix{T}) where T
    out = copy(particles)
    p.poly(out)
    
    function polyLayer_pullback(Δ::T) where T
        newΔ = Array{eltype(Δ)}(undef, length(p.gradient.polynomials[1].coefficients), size(Δ,2))
        @Threads.threads for i in axes(Δ,2)
            newΔ[:,i] .= @views LA.transpose(p.gradient(particles[:,i])) * Δ[:,i]
        end
        
        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), newΔ
    end
    
    return out, polyLayer_pullback
end


function (p::PolyLayer)(particles::Matrix{T}) where T
    return CallPolyLayer(p, particles)
end


function TaylorModel(model::Flux.Chain, z0)
    layers = []
    for i in 1:length(model)
        t = model[i](z0)[:,1]
        cell_taylored = PolyN(t);
        jac_cell_taylored = PolyMN(jacobian(t));
        
        layer = PolyLayer(cell_taylored, jac_cell_taylored)
        push!(layers, layer)
    end
    
    return Flux.Chain(layers...)
end


# end module