import Flux
import TaylorSeries as TS
import LinearAlgebra as LA

"""
Buggy!!!!!!!!
"""
function A(x, y, kn, ks)
    x, y = [TS.TaylorN(1), TS.TaylorN(3)] .+ [x, y]    
    kn = [TS.TaylorN(7), TS.TaylorN(8), TS.TaylorN(9), TS.TaylorN(10)] .+ kn
    ks = [TS.TaylorN(11), TS.TaylorN(12), TS.TaylorN(13), TS.TaylorN(14)] .+ ks

    A = 0.
    for n in 1:length(kn)
        Bn = 1/factorial(n) * kn[n]; An = 1/factorial(n) * ks[n]
        
        A += 1/n * (Bn + im * An) * (x + im * y)^n
    end
        
    return 1 * real(A) / 2.
end


"""
hamiltonian(z, A, h)

Hamiltonian of EM-fields, with z phase space coordinates, A magnetic vector potential and h horizontal curvature.
"""
function hamiltonian(z, A, h)
    x, a, y, b, Δs, δ = [TS.TaylorN(i) for i in 1:length(z)] .+ TS.constant_term(z)
    
    return δ - (1. - h*x) * ( sqrt((1. + δ)^2 - a^2 - b^2) - A)
end


"""
poissonBracket(h, z)

Calculate poisson bracket of h and z.
"""
function poissonBracket(h::TS.TaylorN{T}, z::Vector{TS.TaylorN{T}})::Vector{TS.TaylorN{T}}  where T<:Number
    x, a, y, b, Δs, δ = z
    
    # poisson with x
    x = TS.derivative(x, 1)*TS.derivative(h,2) - TS.derivative(x,2)*TS.derivative(h,1)
    a = TS.derivative(a, 1)*TS.derivative(h,2) - TS.derivative(a,2)*TS.derivative(h,1)
    y = TS.derivative(y, 3)*TS.derivative(h,4) - TS.derivative(y,4)*TS.derivative(h,3)
    b = TS.derivative(b, 3)*TS.derivative(h,4) - TS.derivative(b,4)*TS.derivative(h,3)
    Δs = TS.derivative(Δs, 5)*TS.derivative(h,6) - TS.derivative(Δs,6)*TS.derivative(h,5)
    δ = TS.derivative(δ, 5)*TS.derivative(h,6) - TS.derivative(δ,6)*TS.derivative(h,5)
    
    return -1. * [x, a, y, b, Δs, δ]
end

function poissonBracket(h::TS.TaylorN{T}, z::Vector{TS.TaylorN{T}}, power::Int)::Vector{TS.TaylorN{T}}  where T<:Number   
    for _ in 1:power
        z = poissonBracket(h, z)
    end
    
    return z
end


"""
(z, kn, ks, hx, A; stepsize=1e-2, power=20)

Track z through EM-field specified by kn, ks, A, and curvature hx.
"""
function thick_integration(z, kn, ks, hx; stepsize=1e-2, power::Int=20)
    z_in = [TS.TaylorN(i) for i in 1:length(z)] .+ TS.constant_term(z)
    z = [TS.TaylorN(i) for i in 1:length(z)] .+ TS.constant_term(z)
    
    Apot = A(z[1],z[3],kn,ks)

    # calculate Hamiltonian
    h = hamiltonian(z_in, Apot, hx)
    
    # do poisson
    for i in 1:power
        z_in = poissonBracket(h, z_in)
        z += (-stepsize)^i/factorial(i) * z_in
    end
    
    return z
end


"""
function init_thickMaps(model::Flux.Chain; z=zeros(6))

Taylor beam dynamics around z with zero field strength.
"""
function init_thickMaps(model::Flux.Chain; z::Vector{T}=zeros(6), power::Int=20)::Nothing where T<:Real
    for cell in model
        for element in cell
            element.thickMap = thick_integration(z, zeros(4), zeros(4), 0.; stepsize=element.len, power=power)
        end
    end
end


"""
    propagate(map, z, kn, ks)
    function propagate(element::T, z) where T<:Drift
    function propagate(element::T, z) where T<:Magnet

Propagate phase space coordinates z through beamline element.
"""
function propagate(map, z, kn, ks)
    return [TS.evaluate(map[i], vcat(z, kn, ks)) for i in 1:length(map)]
end

function ChainRulesCore.rrule(::typeof(propagate), map, z, kn, ks)
    z_final = TS.taylor_expand((z)->propagate(map,z[1:6],z[7:10],z[11:end]), vcat(z,kn,ks); order=1)

    function propagate_pullback(Δ)                
        jac = jacobian(z_final)
        newΔ = LA.transpose(jac) * Δ
        
        Δz = newΔ[1:6]
        Δkn = newΔ[7:10]
        Δks = newΔ[11:end]

        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), Δz, Δkn, Δks
    end
    return TS.constant_term(z_final), propagate_pullback
end

function propagate(element::T, z) where T<:Drift
    propagate(element.thickMap, z, zeros(4), zeros(4))
end

function propagate(element::T, z) where T<:Magnet
    propagate(element.thickMap, z, element.kn, element.ks)
end


"""
    jacobian(z)

Calculate TaylorSeries jacobian for z.
"""
function jacobian(z)
    jac = Array{Float64}(undef, length(z), TS.get_numvars())
    
    for i in 1:size(jac)[1]
        for j in 1:size(jac)[2]
            jac[i,j] = TS.derivative(z[i], j) |> TS.constant_term
        end
    end
    
    return jac
end


"""
    plug_in(z::TS.TaylorN{T}, values::Vector{T}, mask_bool::AbstractArray{Bool})::TS.TaylorN{T} where T<:Number
    plug_in(z::AbstractVector{TS.TaylorN{T}}, values::Vector{T}, mask_bool::AbstractArray{Bool})::AbstractVector{TS.TaylorN{T}} where T<:Number

Replace variables specified by mask_bool with values. Requires length(values) == length(mask_bool) == TS.get_numvars()
"""
function plug_in(z::TS.TaylorN{T}, values::Vector{T}, mask_bool::AbstractArray{Bool})::TS.TaylorN{T} where T<:Number
    @assert length(values) == length(mask_bool) == TS.get_numvars() "error: plug in requires all inputs to have equal length to TS.get_numvars()"
    
    base = [TS.TaylorN(i) for i in 1:TS.get_numvars()]
    mask = mask_bool .|> Int
    anti_mask = .!mask_bool .|> Int
    
    newZ = copy(z)
    for order in TS.coeff_table
        for coeff in order
            coeff .* mask |> iszero && continue  ## skip non relevant monomials

            co = TS.getcoeff(z, coeff)
            iszero(co) && continue

            newCo = co * prod((values .^ coeff)[mask_bool])
            newTerm = newCo * prod(((anti_mask .* base) .^ coeff)[.!mask_bool])

            oldTerm = co * prod(base .^ coeff)

            newZ += newTerm - oldTerm
        end
    end
    
    return newZ
end

function plug_in(z::AbstractVector{TS.TaylorN{T}}, values::Vector{T}, mask_bool::AbstractArray{Bool})::AbstractVector{TS.TaylorN{T}} where T<:Number
    [plug_in(z[i], values, mask_bool) for i in eachindex(z)]
end