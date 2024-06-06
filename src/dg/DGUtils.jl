function check_positivity(U, param, discrete_data)
    (; equation) = param
    (; K, Nq) = discrete_data.sizes

    K = num_elements(param)
    rhomin = Inf
    rhoemin = Inf
    rhomax = -Inf
    rhoemax = -Inf
    for k = 1:K
        for i = 1:Nq
            is_positive, rho, p = check_positivity_node(U[i, k], param)
            if !is_positive
                @show k, i, "negative", U[i, k], rho, p
            end
            rhoe = rhoe_ufun(equation, U[i, k])
            rhomin = min(rhomin, rho)
            rhomax = max(rhomax, rho)
            rhoemin = min(rhoemin, rhoe)
            rhoemax = max(rhoemax, rhoe)
        end
    end
    # @show rhomin,rhomax,rhoemin,rhoemax
end

function check_positivity_node(Ui, param)
    (; equation) = param
    is_positive = true
    rho = Ui[1]
    p = pfun(equation, Ui)
    rhoe = rhoe_ufun(equation, Ui)
    if (rho < 0.0 || p < 0.0)
        display("!!!!! Negativity detected !!!!!")
        @show Ui
        is_positive = false
    end
    return is_positive, rho, p
end

function check_conservation(prealloc, param, discrete_data)
    (; Uq) = prealloc
    (; K, Nq) = discrete_data.sizes

    total = zero(prealloc.Uq[1])
    for k = 1:K
        for i = 1:Nq
            total += discrete_data.geom.J[i, k] * discrete_data.ops.wq[i] * Uq[i, k]
        end
    end
    return sum(total)
end
