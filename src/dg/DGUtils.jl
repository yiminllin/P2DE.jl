function check_positivity(U,param)
    @unpack equation = param
    rhomin  = Inf
    rhoemin = Inf
    rhomax  = -Inf
    rhoemax = -Inf
    for k = 1:param.K
        for i = 1:size(U,1)
            is_positive,rho,p = check_positivity_node(U[i,k],param)
            if !is_positive
                @show k,i,"negative",U[i,k],rho,p
            end
            rhoe  = rhoe_ufun(equation,U[i,k])
            rhomin = min(rhomin,rho)
            rhomax = max(rhomax,rho)
            rhoemin = min(rhoemin,rhoe)
            rhoemax = max(rhoemax,rhoe)
        end
    end
    # @show rhomin,rhomax,rhoemin,rhoemax
end

function check_positivity_node(Ui,param)
    @unpack equation = param
    is_positive = true 
    rho  = Ui[1]
    p    = pfun(equation,Ui)
    rhoe = rhoe_ufun(equation,Ui)
    if (rho < 0.0 || p < 0.0)
        display("!!!!! Negativity detected !!!!!")
        @show Ui
        is_positive = false
    end
    return is_positive,rho,p
end

function check_conservation(prealloc,param,discrete_data_gauss,discrete_data_LGL)
    @unpack Uq,LGLind = prealloc
    total = zero(SVector{3,Float64})   # TODO: hardcoded
    for k = 1:param.K
        for i = 1:size(Uq,1)
            total += LGLind[k] ? discrete_data_LGL.geom.J[i,k]*discrete_data_LGL.ops.wq[i]*Uq[i,k] : discrete_data_gauss.geom.J[i,k]*discrete_data_gauss.ops.wq[i]*Uq[i,k]
        end
    end
    return sum(total)
end