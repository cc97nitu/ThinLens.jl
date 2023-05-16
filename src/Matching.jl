import Optim
import TaylorSeries as TS


function symbolicMultipoles(model::S; k1f::T=0.,k1d::T=0.,k2f::T=0.,k2d::T=0.) where {S,T}
    model_tpsa = deepcopy(model)
    
    # tpsa vectors
    k1f_tpsa = [0,TS.TaylorN(8) + k1f,0,0]
    k1d_tpsa = [0,TS.TaylorN(9) + k1d,0,0]
    k2f_tpsa = [0,0,TS.TaylorN(10) + k2f,0]
    k2d_tpsa = [0,0,TS.TaylorN(11) + k2d,0]

    # update model
    for i in 1:length(model_tpsa)
        for j in 1:length(model_tpsa[i])
            if typeof(model_tpsa[i][j]) == TL.Quadrupole
                if model_tpsa[i][j].kn[2] >= 0.
                    model_tpsa[i][j].kn = k1f_tpsa
                else
                    model_tpsa[i][j].kn = k1d_tpsa
                end
                
                model_tpsa[i][j].ks = zeros(eltype(k1f_tpsa),4)
            elseif typeof(model[i][j]) == TL.Sextupole
                if model_tpsa[i][j].kn[3] >= 0.
                    model_tpsa[i][j].kn = k2f_tpsa
                else
                    model_tpsa[i][j].kn = k2d_tpsa
                end
                
                model_tpsa[i][j].ks = zeros(eltype(k1f_tpsa),4)
            end
        end
    end
    
    return model_tpsa, k1f_tpsa, k1d_tpsa, k2f_tpsa, k2d_tpsa
end

function tunesChromas_tpsa(model)
    origin_tpsa = [TS.TaylorN(i) for i in 1:7] + [0,0,0,0,0,0,1.]

    tracked = model(origin_tpsa)

    # trace of one-turn map
    horTrace = TS.derivative(tracked[1,1],1) + TS.derivative(tracked[2,1],2)
    verTrace = TS.derivative(tracked[3,1],3) + TS.derivative(tracked[4,1],4)

    q1 = acos(1/2 * horTrace) / (2π)
    q2 = acos(1/2 * verTrace) / (2π)

    dq1 = TS.derivative(q1, 6)
    dq2 = TS.derivative(q2, 6)
    return q1, q2, dq1, dq2
end

function mse_tunes(model_tpsa, k1f_tpsa, k1d_tpsa, k1f, k1d, q1_target, q2_target)
    # update model in place
    k1f_tpsa[2] -= TS.constant_term(k1f_tpsa[2])
    k1f_tpsa[2] += k1f
    
    k1d_tpsa[2] -= TS.constant_term(k1d_tpsa[2])
    k1d_tpsa[2] += k1d
    
    q1, q2, _, _ = tunesChromas_tpsa(model_tpsa)
    
    obj = 1/2 * ((q1 - q1_target)^2 + (q2 - q2_target)^2)
    
    obj_∂k1f = TS.derivative(obj, 8) # ∂/∂k1f
    obj_∂k1d = TS.derivative(obj, 9) # ∂/∂k1d

    return TS.constant_term(obj), TS.constant_term(obj_∂k1f), TS.constant_term(obj_∂k1d)
end

function matchTunes(model, q1_target, q2_target; k1f_start=0.35, k1d_start=-0.309)
    # keep old TaylorSeries configuration
    old_vars = TS.get_variable_names()
    old_order = TS.get_order()

    TS.set_variables("x a y b σ δ β0β k1f k1d k2f k2d", order=2)

    # optimize w.r.t. tune
    model_tpsa, k1f_tpsa, k1d_tpsa, k2f_tpsa, k2d_tpsa = symbolicMultipoles(model; k1f=k1f_start, k1d=k1d_start)
    
    obj(x) = mse_tunes(model_tpsa, k1f_tpsa, k1d_tpsa, x[1], x[2], q1_target, q2_target)[1]
    function der_obj!(G, x)
        _, ∂k1f, ∂k1d = mse_tunes(model_tpsa, k1f_tpsa, k1d_tpsa, x[1], x[2], q1_target, q2_target)
        G[1] = ∂k1f
        G[2] = ∂k1d
    end
    
    res = Optim.optimize(obj, der_obj!, [k1f_start, k1d_start], Optim.LBFGS())
    
    # restore previous TaylorSeries configuration
    TS.set_variables(old_vars, order=old_order)

    return res
end

function mse_tunesChromas(model_tpsa, k1f_tpsa, k1d_tpsa, k1f, k1d, k2f_tpsa, k2d_tpsa, k2f, k2d, q1_target, q2_target, dq1_target, dq2_target)
    # update model in place
    k1f_tpsa[2] -= TS.constant_term(k1f_tpsa[2])
    k1f_tpsa[2] += k1f
    
    k1d_tpsa[2] -= TS.constant_term(k1d_tpsa[2])
    k1d_tpsa[2] += k1d
    
    k2f_tpsa[3] -= TS.constant_term(k2f_tpsa[3])
    k2f_tpsa[3] += k2f
    
    k2d_tpsa[3] -= TS.constant_term(k2d_tpsa[3])
    k2d_tpsa[3] += k2d

    q1, q2, dq1, dq2 = tunesChromas_tpsa(model_tpsa)
    
    obj1 = 1/4 * ((q1 - q1_target)^2 + (q2 - q2_target)^2)
    obj2 = 1/4 * ((dq1 - dq1_target)^2 + (dq2 - dq2_target)^2)
    obj = sqrt(obj2)

    obj_∂k1f = TS.derivative(obj1, 8) # ∂/∂k1f
    obj_∂k1d = TS.derivative(obj1, 9) # ∂/∂k1d
    obj_∂k2f = TS.derivative(obj2, 10) # ∂/∂k1f
    obj_∂k2d = TS.derivative(obj2, 11) # ∂/∂k1f

    return TS.constant_term(obj), TS.constant_term(obj_∂k1f), TS.constant_term(obj_∂k2f), TS.constant_term(obj_∂k2f), TS.constant_term(obj_∂k2d)
end

function matchTunesChromas(model, q1_target, q2_target, dq1_target, dq2_target; k1f_start=0.35, k1d_start=-0.309)
    # keep old TaylorSeries configuration
    old_vars = TS.get_variable_names()
    old_order = TS.get_order()

    TS.set_variables("x a y b σ δ β0β k1f k1d k2f k2d", order=3)

    # optimize w.r.t. tune
    model_tpsa, k1f_tpsa, k1d_tpsa, k2f_tpsa, k2d_tpsa = symbolicMultipoles(model; k1f=k1f_start, k1d=k1d_start)
    
    obj(x) = mse_tunesChromas(model_tpsa, k1f_tpsa, k1d_tpsa, x[1], x[2], k2f_tpsa, k2d_tpsa, x[3], x[4], q1_target, q2_target, dq1_target, dq2_target)[1]
    function der_obj!(G, x)
        _, ∂k1f, ∂k1d, ∂k2f, ∂k2d = mse_tunesChromas(model_tpsa, k1f_tpsa, k1d_tpsa, x[1], x[2], k2f_tpsa, k2d_tpsa, x[3], x[4], q1_target, q2_target, dq1_target, dq2_target)
        G[1] = ∂k1f
        G[2] = ∂k1d
        G[3] = ∂k2f
        G[4] = ∂k2d
    end
    
    res = Optim.optimize(obj, der_obj!, [k1f_start, k1d_start, 0, 0], Optim.LBFGS())
    
    # restore previous TaylorSeries configuration
    TS.set_variables(old_vars, order=old_order)

    return res
end

function mse_chromas(model_tpsa, k2f_tpsa, k2d_tpsa, k2f, k2d, dq1_target, dq2_target)
    # update model in place
    k2f_tpsa[3] -= TS.constant_term(k2f_tpsa[3])
    k2f_tpsa[3] += k2f
    
    k2d_tpsa[3] -= TS.constant_term(k2d_tpsa[3])
    k2d_tpsa[3] += k2d

    _, _, dq1, dq2 = tunesChromas_tpsa(model_tpsa)
    
    obj2 = (dq1 - dq1_target)^2 + (dq2 - dq2_target)^2
    obj = sqrt(obj2)

    obj_∂k2f = TS.derivative(obj, 10) # ∂/∂k1f
    obj_∂k2d = TS.derivative(obj, 11) # ∂/∂k1f

    return TS.constant_term(obj), TS.constant_term(obj_∂k2f), TS.constant_term(obj_∂k2d)
end

function matchChromas(model, dq1_target, dq2_target; k1f_start=0.35, k1d_start=-0.309)
    # keep old TaylorSeries configuration
    old_vars = TS.get_variable_names()
    old_order = TS.get_order()

    TS.set_variables("x a y b σ δ β0β k1f k1d k2f k2d", order=3)

    # optimize w.r.t. tune
    model_tpsa, k1f_tpsa, _, k2f_tpsa, k2d_tpsa = symbolicMultipoles(model; k1f=k1f_start, k1d=k1d_start)
    
    obj(x) = mse_chromas(model_tpsa, k2f_tpsa, k2d_tpsa, x[1], x[2], dq1_target, dq2_target)[1]
    function der_obj!(G, x)
        _, ∂k2f, ∂k2d = mse_chromas(model_tpsa, k2f_tpsa, k2d_tpsa, x[1], x[2], dq1_target, dq2_target)
        G[1] = ∂k2f
        G[2] = ∂k2d
    end
    
    res = Optim.optimize(obj, der_obj!, zeros(2), Optim.LBFGS())
    
    # restore previous TaylorSeries configuration
    TS.set_variables(old_vars, order=old_order)

    return res
end