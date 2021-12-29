# set up test suite
particle = [-8.16e-03, -1.78e-03,  5.55e-03,  1.93e-03,  0.00e+00,  7.87e-04, 9.99e-01]

fdm = FD.central_fdm(5, 1)

function compareJacobians(jacs1, jacs2)
    for (i, (j1, j2)) in enumerate(zip(jacs1, jacs2))
        diff = j1 .- j2
        
        if any(x -> abs(x)>1e-9, diff)
            display(diff)
            println("\nerror at gradient $i")
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

# test thinMultipole
wrap_thinMultipole(p, kn, ks) = ThinLens.thinMultipole(p, 0.1, kn, ks)

kn = [0.1, 0.5, -0.2]
ks = [-0.05, 0.3, 0.8]

jaxFD_thinMultipole = FD.jacobian(fdm, wrap_thinMultipole, particle, kn, ks)
jaxFlux_thinMultipole = Flux.jacobian(wrap_thinMultipole, particle, kn, ks)

@test compareJacobians(jaxFD_thinMultipole, jaxFlux_thinMultipole)

# test curved thinMultipole
wrap_curvedThinMultipole(p, kn, ks) = ThinLens.thinMultipole(p, 0.1, kn, ks, 0.1, 0.)

kn = [0.1, 0.5, -0.2]
ks = [-0.05, 0.3, 0.8]

jaxFD_curvedThinMultipole = FD.jacobian(fdm, wrap_curvedThinMultipole, particle, kn, ks)
jaxFlux_curvedThinMultipole = Flux.jacobian(wrap_curvedThinMultipole, particle, kn, ks)

@test compareJacobians(jaxFD_curvedThinMultipole, jaxFlux_curvedThinMultipole)