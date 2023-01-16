# module Polynomial

import TaylorSeries as TS

# export PolyN


struct PolyN{S,T}
    coefficients::Vector{Vector{S}}
    coeffTable::Vector{Vector{Vector{T}}}
end


function PolyN(t::Vector{TS.TaylorN{T}}) where T<:Number
    coeffTable = TS.coeff_table
    noCoeff = coeffTable .|> length |> sum
    
    coefficients = [Vector{TS.numtype(t[1])}(undef, noCoeff) for _ in eachindex(t)]
    
    for n in eachindex(coefficients)
        c = 1
        for i in eachindex(coeffTable)
            for j in eachindex(coeffTable[i])
                coefficients[n][c] = TS.getcoeff(t[n], coeffTable[i][j])

                c += 1
            end
        end
    end
    
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


struct PolyMN
    polynomials::Array{PolyN, 1}
end


function PolyMN(m::Matrix{TaylorSeries.TaylorN{S}}) where S<:Number
    polynomials = Vector{PolyN}(undef, size(m)[1])

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


# end  # module