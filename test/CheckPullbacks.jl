# set up test suite
particle = [-8.16e-03, -1.78e-03,  5.55e-03,  1.93e-03,  0.00e+00,  7.87e-04, 9.99e-01]

fdm = FD.central_fdm(5, 1)

function compareJacobians(jacs1, jacs2)
    for (j1, j2) in zip(jacs1, jacs2)
        diff = j1 .- j2
        
        if any(x -> abs(x)>1e-9, diff)
            display(diff)
            return false
        end
    end

    return true
end

# test linear drift
wrap_driftLinear(p) = ThinLens.driftLinear(p, 3.0)

jaxFD_drift = FD.jacobian(fdm, wrap_driftLinear, particle)
jaxFlux_drift = Flux.jacobian(wrap_driftLinear, particle)

@test compareJacobians(jaxFD_drift, jaxFlux_drift)

# test exact drift
wrap_driftExact(p) = ThinLens.driftExact(p, 3.)

jaxFD_driftExact = FD.jacobian(fdm, wrap_driftExact, particle)
jaxFlux_driftExact = Flux.jacobian(wrap_driftExact, particle)

@test compareJacobians(jaxFD_driftExact, jaxFlux_driftExact)


# test tM_2ndOrder
wrap_tM_2ndOrder(p, kn, ks) = ThinLens.tM_2ndOrder(p, 10., kn, ks)
kn = [0.1, 0.5, -0.2]
ks = [-0.05, 0.3, 0.8]

jaxFD_tM_2ndOrder = FD.jacobian(fdm, wrap_tM_2ndOrder, particle, kn, ks)
jaxFlux_tM_2ndOrder = Flux.jacobian(wrap_tM_2ndOrder, particle, kn, ks)

@test compareJacobians(jaxFD_tM_2ndOrder, jaxFlux_tM_2ndOrder)

# test curvature Kick
wrap_curvatureEffectKick(p, kn, ks) = ThinLens.curvatureEffectKick(p, 0.32, kn, ks, 0.1, 0.)
kn = [0.1, 0.5, -0.2]
ks = [0.0, 0.3, 0.8]

jaxFD_tM_curvatureEffectKick = FD.jacobian(fdm, wrap_tM_2ndOrder, particle, kn, ks)
jaxFlux_tM_curvatureEffectKick = Flux.jacobian(wrap_tM_2ndOrder, particle, kn, ks)

@test compareJacobians(jaxFD_tM_curvatureEffectKick, jaxFlux_tM_curvatureEffectKick)




#########
wrap_tM(p) = ThinLens.tM(p, 3.0, kn, ks)

jaxFD_tM = FD.jacobian(fdm, wrap_tM, particle)
jaxFlux_tM = Flux.jacobian(wrap_tM, particle)

@test compareJacobians(jaxFD_tM, jaxFlux_tM)
